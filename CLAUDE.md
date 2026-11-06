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
- **The P2 plan and ledger:** [docs/superpowers/plans/2026-11-06-p2-input-preflight.md](docs/superpowers/plans/2026-11-06-p2-input-preflight.md), and `.superpowers/sdd/2026-11-06-p2-input-preflight/progress.md` (git-ignored, local). The plan's "Scope rulings" say which deferred items stay deferred and to which sub-project they belong.
- **The user-facing behaviour:** [USER_MANUAL.md](USER_MANUAL.md).

## Rules that always apply
- **Clean room.** SHARC 4.1 (installed at `\\wsl.localhost\OracleLinux_9_5\home\ajc\sharc4-4.1`) is a reference for the workflow and a black box for output comparison. Implement methods from the literature. SHARC may only be *executed*, from `tools/sharc_compare/`, and must never be imported by `gqteashmd`. The licence is MIT ("The gqteashmd developers").
- **Run everything in WSL.** The repository lives at `C:\Users\ajc\OneDrive\gqteaSHMD` (`/mnt/c/Users/ajc/OneDrive/gqteaSHMD` in WSL), but code, tests and git run inside WSL Oracle Linux 9.5. From the Windows host, wrap each command:
  ```bash
  wsl -d OracleLinux_9_5 -- bash -lc 'cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD && conda activate gqteashmd-dev && <command>'
  ```
  Edit files with the normal file tools, using LF line endings.
- **Git from WSL only.** Pre-commit hooks (ruff 0.17.4) are installed in WSL and do not run from Windows Git. Pushing from Windows is fine (`git -C C:\Users\ajc\OneDrive\gqteaSHMD push ...`), because Windows has the GitHub credentials and pushing runs no hooks. Never use `--no-verify`.
- **Keep project runs, conda environments and large outputs out of OneDrive and out of `/mnt/c`.** Use `~/...` or `/tmp` in WSL. pytest's `tmp_path` already does this.
- **Workflow.** Feature branch per sub-project (`feat/<topic>`), one commit per task, conventional commit subjects (`feat:`, `fix:`, `test:`, `docs:`, `ci:`, `build:`). The owner decides merges. P1 was merged locally into `main`, and `main` is not yet pushed to GitHub (`AdemirCamargo57/gqteaSHMD`).
- **Commit trailer.** End each commit message with the `Co-Authored-By:` trailer for the model that wrote the commit.
- **Keep the two documents current.** Update **CLAUDE.md and USER_MANUAL.md in the same branch as any change to behaviour, inputs, outputs, formulas or status.** USER_MANUAL.md must always mark what is implemented and what is planned.

## Environment
- Conda env `gqteashmd-dev` (Miniforge in WSL), created from `environment.yml`. The package is installed editable with `pip install -e . --no-deps`. Python 3.12.
- The runtime dependency is `pydantic>=2.6` only. PySCF, NumPy, SciPy, h5py and geomeTRIC are preinstalled for P3+.
- Tests:
  - `pytest -m "not slow"` takes about 1 min.
  - `pytest -W error` runs everything, including the slow kill/resume tests, in about 3 min.
  - The tests start real subprocesses, so they need Linux.
- Lint: `ruff format . && ruff check .`. The line length is 100, with rules E, F, W, I, B, UP and SIM. Markdown is excluded (`extend-exclude`), because ruff ≥0.17 would rewrite code blocks in the plans.
- Types: `mypy` runs clean on `src/` (configuration in `pyproject.toml`) and is part of CI.
- CI: `.github/workflows/ci.yml` runs in an `oraclelinux:9` container on push, on PRs and nightly. It has never run, because `main` has not been pushed yet.
- Workstation facts in WSL: 16 cores and about 7 GB RAM. ORCA 6.1.1 and Gaussian 16 are installed, but they are not used before phase P7.

## Architecture (implemented in P1 and P2)
Dependencies point down only; domain code never imports stages or the orchestrator.

```text
cli.py ─► orchestrator.Orchestrator ─► execution.local.LocalExecutor (one per stage) ─► subprocess: python -m gqteashmd.worker task.json
   │         │  preflight.py (before the project directory is touched)       worker ─► stage.run_task() ─► result.json
   │         │  project/state.py (SQLite, single writer), project/provenance.py
   │         │  project/layout.py, project/lock.py (flock)
   │         └► stages/ (Stage contract: plan → run_task → finalize) ─► chem/, util/, (later) electronic/, vib/, dynamics/
   └► logging_setup.py (console INFO, logs/gqteashmd.log DEBUG)
```

- **Stage contract** (`stages/base.py`):
  - `plan(ctx)` runs in the controller and returns `TaskSpec` objects deterministically.
  - `run_task(ctx, task, task_dir)` runs in a worker subprocess. It must be idempotent, write only in `task_dir`, and return a small JSON-serialisable dict.
  - `finalize(ctx, results)` runs in the controller, writes the stage artifacts and raises `StageFailedError` if checks fail.
  - Stage attributes: `name`, `version`, `depends_on`, `config_sections`, `max_failed_fraction`, `enabled(config)`.
- **Worker protocol** (`worker.py`): the worker reads `task.json` and writes `result.json` atomically, either `{"status": "ok", "result": ...}` or `{"status": "error", "error_type", "message", "retryable", "traceback"?}`. Errors are also written to stderr, which goes to `worker.log`. A worker that exits without `result.json` is treated as a crash, which is retryable. Workers read the config snapshot `.gqteashmd/config.json` and never the user's input file.
- **Executor** (`execution/local.py`):
  - one subprocess per task, each in its own session;
  - the orchestrator builds **one executor per stage** from that stage's effective resources (`ResourcesSection.for_stage(name)`: a `[resources.stages.<name>]` key, or the global value if unset);
  - at most `max_parallel_tasks` running at once;
  - sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to `threads_per_task`, and `PYSCF_MAX_MEMORY` to `memory_mb_per_task`;
  - on stop: SIGTERM to the process group, SIGKILL after `stop_grace_period_s`;
  - if a callback raises, it reaps every worker before re-raising.
- **Orchestrator rules** (`orchestrator.py`, spec §3.4):
  - **Fingerprint:** SHA-256 over the stage name, version, `fingerprint_inputs()` and the upstream fingerprints. A DONE stage with the same fingerprint is skipped.
  - **Changed fingerprint:** a stage that is DONE or RUNNING, **or that has any DONE task**, raises `StaleStageError`, which tells the user to run `--rerun-from STAGE`. Only stages with no finished task are reset automatically.
  - **`--rerun-from`:** moves the old stage directories to `.gqteashmd/superseded/`, never deleting them. It first terminates orphaned workers.
  - **Retries:** a crash or a retryable error is retried while `attempts < max_attempts`. An interrupted task is not counted as an attempt.
  - **Stage failure:** a stage fails when `failed/total > max_failed_fraction`. Any exception inside a stage marks it FAILED.
  - **Restart:** tasks left RUNNING by a dead controller are reconciled. A task with an ok `result.json` becomes DONE; otherwise its orphan is killed (verified via `/proc/<pid>/cmdline`) and the task is requeued.
  - **`--rerun-from`** is applied once per orchestrator (`self.rerun_from` is cleared after use). The hint printed for a stale stage is shell-quoted. A fingerprint input that is not JSON-serializable raises `ProjectError`.
  - **Pause:** after a stage listed in `workflow.pause_after` completes **in this run** and is not the last stage, `RunSummary.paused_after` is set, the loop stops and the CLI prints a message and exits 5 (`EXIT_STOPPED`). A rerun continues, because that stage is then up to date. A skipped or up-to-date stage never pauses. Unknown names in `pause_after` and in `[resources.stages]` are a `ConfigError` raised in `Orchestrator.__init__`.
- **Config** (`config/schema.py`): `[resources]` requires `total_cores` and `total_memory_mb`. A model validator checks `max_parallel_tasks × threads_per_task ≤ total_cores` and `max_parallel_tasks × memory_mb_per_task ≤ total_memory_mb` for the global values and for every stage override (merged with the global values). `execution.min_free_disk_mb` (default 1000) and `[workflow] pause_after` also live here. `loader.py` expands `~` (an unknown `~user` is a `ConfigError`) and resolves relative paths against the input file's directory. The later input sections are not in the schema yet (they arrive with P3–P8).
- **Preflight** (`preflight.py:run_preflight`): runs from `Orchestrator.preflight()` (called by `cli._cmd_run`; the result is cached, so `run()` does not repeat it), **before** `layout.ensure()` creates anything.
  - It raises `ConfigError` for: a project directory on a Windows drive in WSL (`/mnt/<letter>/`; `/mnt/data` is fine) or under a OneDrive folder; free disk below `min_free_disk_mb` or a failed disk query; a missing `required_modules` entry of an enabled stage.
  - It only returns a **warning** (D17) when `total_cores` or `total_memory_mb` exceeds the detected machine (`detect_machine()`: `sched_getaffinity` and `/proc/meminfo`). The CLI logs warnings after logging is configured, so they reach the console and the log file.
  - The location check is on the path text only; a symlink is not followed.
- **Provenance** (`project/provenance.py`): each `run` that passes preflight appends one record to `.gqteashmd/provenance.json` (`started_at`, `gqteashmd`, `git_commit`, `git_dirty`, `python`, `platform`, `hostname`, `cpu_count`, `packages`). A damaged file with invalid JSON is replaced by a fresh list; non-UTF-8 bytes are not handled (see limitations).
- **State** (`project/state.py`): `StateStore(db_path, *, readonly=False)`. Any `sqlite3.DatabaseError` (including a db path that is a directory) becomes `ProjectError` ("unreadable or corrupt", exit 3). `status` opens the db read-only (`mode=ro`) and never writes.
- **Lock** (`project/lock.py`): `ProjectLock` tries `flock` twice, 0.2 s apart (`locked_by` probes the same lock briefly). The error reads "pid unknown" when the holder's PID cannot be read.
- **Logging** (`logging_setup.py`): `configure_logging` is called by `run` after the project directory exists; `close_logging` (in `cli.main`'s `finally`) closes all handlers and returns to a `NullHandler`. Records logged with `extra=FILE_ONLY` go to `logs/gqteashmd.log` but not to the console. `cli.main` uses it for errors and tracebacks, so each error is printed **once** on stderr (`gqteashmd: error: <message>`) with no traceback.
- **Errors** (`errors.py`):
  - Exit codes: `ShmdError` 1, `ConfigError` 2, `ProjectError` 3 (`StaleStageError` is a subclass), `StageFailedError` 4. Stopped by the user **or paused for review** is 5 (`cli.EXIT_STOPPED`).
  - `TaskError(msg, retryable=...)` is raised inside tasks.
  - Every error reaches the user as `gqteashmd: error: <message>`.
  - `.` and `..` are not safe stage or task names (`is_safe_name`). The XYZ parser rejects non-finite coordinates; `validate` also requires the geometry to be a regular file.
- **Testing seam:** `gqteashmd.testing.fake_stages` provides fake stages whose behaviour is set by the environment variable `GQTEASHMD_FAKE_BEHAVIOUR` (JSON). The keys are `sleep`, `fail`, `unexpected`, `crash_once`, `retryable_once`, `plan_raises` and `skip_optimize`. The CLI's hidden `--pipeline` option selects them in tests.
- **The production pipeline** (`stages/registry.py:default_pipeline`) contains only `prepare` today.

## Key decisions (full list in master plan §10)
- **Python core.** Electronic structure dominates the cost, so the core is plain Python (D1).
- **PySCF only first.** PySCF is the only backend in the first release; ORCA, Gaussian, OpenMolcas and OpenQP come later behind the same backend contract (D4).
- **Single workstation.** The resumable controller runs one subprocess per task, with no scheduler (D5).
- **Strict input file.** One strict TOML input. Units go in key names (`dt_fs`, `temperature_K`, ...), and state labels are `S0`, `S1`, `T1` (0-based internally) (D7, D13).
- **Hardware from the user.** The user enters the hardware in `[resources]`. The program warns, without stopping, when the values exceed the detected machine (D17, implemented in P2).
- **Naming.** The package and command are `gqteashmd`; internal class names keep the `Shmd` prefix (D18).
- **P3 validation target.** Ethylene (`C:\AJCamargo\TOPICOS\Eteno\Eteno.xyz` → `tests/data/ethylene.xyz`) at SA2-CASSCF(2,2)/6-31G\* first, then TDA (D19).

## Current status (2026-11-06)
- **P1 — foundation & orchestrator: DONE.** It is merged into local `main` at `e4a9319` (111 tests passing) and not pushed.
- **P2 — input schema & preflight: DONE, on branch `feat/p2-input-preflight`, NOT yet merged** (owner decides the merge; `main` is not pushed). Full suite: 168 tests passing (`pytest -W error`), ruff and mypy clean.
- **Implemented features:**
  - the CLI commands `init`, `validate`, `run [--rerun-from]`, `status` and `stop`;
  - the input sections `[project]`, `[molecule]`, `[resources]` (with required `total_cores` and `total_memory_mb`, and `[resources.stages.<name>]` overrides), `[execution]` (with `min_free_disk_mb`) and `[workflow]` (`pause_after`);
  - preflight checks before a run, provenance record per run, review pauses (exit 5), per-stage executors;
  - the `prepare` stage, which validates the XYZ file, the elements and the charge/multiplicity parity, and writes `geometry.xyz` and `molecule.json`;
  - resumable, crash-safe orchestration.
- **Not implemented yet:** every quantum-chemistry stage (optimize, frequencies, sampling, excitations, selection, dynamics, analysis), the PySCF backend, and the input sections of those stages (`[electronic]`, `[optimize]`, `[frequencies]`, `[sampling]`, `[excitations]`, `[selection]`, `[dynamics]`, `[analysis]`). By ruling, each sub-project adds the input sections it consumes (their keys depend on open items O3, O4, O8, O19). State labels (`S0`/`S1`/`T1`) arrive with `[electronic]` in P3.

## Known limitations
1. **Only `prepare` exists.** There is no quantum chemistry yet, so per-stage resources and `pause_after` have no visible effect (the last stage never pauses).
2. **The project-folder check is a path-text check.** An absolute path that is a symlink into a Windows drive or a OneDrive folder is not detected (relative paths are resolved, so they are).
3. **A corrupt `provenance.json` with non-UTF-8 bytes would stop a run** (invalid JSON is tolerated and replaced). gqteashmd writes that file itself, so this is unlikely.
4. **CI has never run**, because `main` has not been pushed.
5. **Other deferred items** are listed in the P1 and P2 ledgers under `.superpowers/sdd/` (git-ignored, local). The P2 plan's "Scope rulings" assign them: refused runs still writing state, and optional-stage handling (P4); the worker orphan window between `Popen` and `on_start` (P7b); SIGKILL to the group after the leader exits, and an unreadable `task.json` (P3); `.tmp` cleanup, a DONE stage directory deleted by hand, and `ensure()`/logging set up before the lock (no sub-project assigned).

## Remaining work (master plan §9)
**P3 is next.**

| ID | Sub-project | Notes |
|---|---|---|
| P2 | Input schema & preflight | **DONE** on `feat/p2-input-preflight` (not merged); see Status |
| P3 | PySCF electronic-structure layer | **Next.** Adds `[electronic]` and state labels. Backend contract (strategy §2.3); SCF/TDA/TDDFT/SA-CASSCF energies, gradients, NACs, Hessian; ethylene validation |
| P4 | `optimize` + `frequencies` stages | geomeTRIC; own harmonic analysis; Molden output |
| P5 | `sampling` stage | Own Wigner sampler; compare with SHARC `wigner.py` |
| P6 | `excitations` + `selection` stages | Spectrum and initial-state selection |
| P7a/b | Dynamics core (models), then PySCF dynamics stage | Strategy §4 |
| P8 | `analysis` stage + end-to-end test | Populations and kinetics |
| P9 | SHARC validation campaign | Strategy §5 |

**Process for each sub-project:**
1. Write the detailed plan with superpowers:writing-plans in `docs/superpowers/plans/`.
2. Execute it with superpowers:subagent-driven-development. The ledger goes in `.superpowers/sdd/<plan>/progress.md`; trust the ledger and `git log` after context loss.
3. Run the final review.
4. Finish with superpowers:finishing-a-development-branch.
