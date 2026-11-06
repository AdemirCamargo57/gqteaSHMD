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
- **The P1 ledger:** `.superpowers/sdd/2026-11-06-p1-foundation-orchestrator/progress.md`. It is git-ignored and local to the development machine. It contains every ruling made during P1 and the deferred findings that make up the P2/P3 checklist.
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
- CI: `.github/workflows/ci.yml` runs in an `oraclelinux:9` container on push, on PRs and nightly. It has never run, because `main` has not been pushed yet.
- Workstation facts in WSL: 16 cores and about 7 GB RAM. ORCA 6.1.1 and Gaussian 16 are installed, but they are not used before phase P7.

## Architecture (implemented in P1)
Dependencies point down only; domain code never imports stages or the orchestrator.

```text
cli.py ─► orchestrator.Orchestrator ─► execution.local.LocalExecutor ─► subprocess: python -m gqteashmd.worker task.json
             │  project/state.py (SQLite, single writer)          worker ─► stage.run_task() ─► result.json
             │  project/layout.py, project/lock.py (flock)
             └► stages/ (Stage contract: plan → run_task → finalize) ─► chem/, util/, (later) electronic/, vib/, dynamics/
```

- **Stage contract** (`stages/base.py`):
  - `plan(ctx)` runs in the controller and returns `TaskSpec` objects deterministically.
  - `run_task(ctx, task, task_dir)` runs in a worker subprocess. It must be idempotent, write only in `task_dir`, and return a small JSON-serialisable dict.
  - `finalize(ctx, results)` runs in the controller, writes the stage artifacts and raises `StageFailedError` if checks fail.
  - Stage attributes: `name`, `version`, `depends_on`, `config_sections`, `max_failed_fraction`, `enabled(config)`.
- **Worker protocol** (`worker.py`): the worker reads `task.json` and writes `result.json` atomically, either `{"status": "ok", "result": ...}` or `{"status": "error", "error_type", "message", "retryable", "traceback"?}`. Errors are also written to stderr, which goes to `worker.log`. A worker that exits without `result.json` is treated as a crash, which is retryable. Workers read the config snapshot `.gqteashmd/config.json` and never the user's input file.
- **Executor** (`execution/local.py`):
  - one subprocess per task, each in its own session;
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
- **Errors** (`errors.py`):
  - Exit codes: `ShmdError` 1, `ConfigError` 2, `ProjectError` 3 (`StaleStageError` is a subclass), `StageFailedError` 4. Stopped by the user is 5 (`cli.EXIT_STOPPED`).
  - `TaskError(msg, retryable=...)` is raised inside tasks.
  - Every error reaches the user as `gqteashmd: error: <message>`.
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
- **P1 — foundation & orchestrator: DONE.** It is merged into local `main` at `e4a9319` (111 tests passing) and not pushed. The working branch is now `feat/p2-input-preflight`.
- **Implemented features:**
  - the CLI commands `init`, `validate`, `run [--rerun-from]`, `status` and `stop`;
  - the input sections `[project]`, `[molecule]`, `[resources]` and `[execution]`;
  - the `prepare` stage, which validates the XYZ file, the elements and the charge/multiplicity parity, and writes `geometry.xyz` and `molecule.json`;
  - resumable, crash-safe orchestration.
- **Not implemented yet:** every quantum-chemistry stage (optimize, frequencies, sampling, excitations, selection, dynamics, analysis) and the PySCF backend.

## Known limitations (fix first in P2)
1. **Every error prints twice.** `cli.main()` logs each `ShmdError`, so the message appears twice on stderr (once from the logging fallback or the INFO console handler, once from the print). An internal error also prints a raw traceback before the "internal error" line. Fix: add a `NullHandler` on `gqteashmd`, log in `main()` at DEBUG with `exc_info`, and add a subprocess test that asserts a single message and no "Traceback".
2. **A bad state-db path gives the wrong error.** `StateStore.__init__` calls `sqlite3.connect` outside its `try`, so a db path that is a directory gives "internal error" (exit 1) instead of `ProjectError` (exit 3).
3. **More deferred items are in the P1 ledger.** It lists about 40 smaller findings. The ones marked "P2" form the P2 opening checklist, for example: `status` writes to the db, nan/inf coordinates are accepted, `.`/`..` pass the safe-name check, the `locked_by` probe race, and logging handlers are never closed.

## Remaining work (master plan §9)
| ID | Sub-project | Notes |
|---|---|---|
| P2 | Input schema & preflight | Limitations 1–2 above, the P2 items in the ledger, the hardware keys (D17), per-stage resources, `[workflow] pause_after`, state labels, provenance, mypy |
| P3 | PySCF electronic-structure layer | Backend contract (strategy §2.3); SCF/TDA/TDDFT/SA-CASSCF energies, gradients, NACs, Hessian; ethylene validation |
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
