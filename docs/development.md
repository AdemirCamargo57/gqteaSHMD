# Developer guide

## Where things live
- Source: `C:\Users\ajc\OneDrive\gqteaSHMD` = `/mnt/c/Users/ajc/OneDrive/gqteaSHMD` in WSL.
- Run everything (tests, gqteashmd) **inside WSL (Oracle Linux 9.5)** or on the Oracle Linux workstation.
  gqteashmd is Linux-only (PySCF, fcntl, /proc).
- Never put conda environments, gqteashmd project runs or large outputs in the OneDrive folder.
  Use `~/gqteashmd-runs/` (WSL) or a local data disk on the workstation.

## One-time setup (WSL)
```bash
# This WSL has Miniforge conda. (micromamba works the same way: replace `conda` by `micromamba`.)
cd /mnt/c/Users/ajc/OneDrive/gqteaSHMD
conda env create -f environment.yml
conda activate gqteashmd-dev
pip install -e . --no-deps
pre-commit install
git config core.autocrlf false
git config core.filemode false
```
After a change of `__version__` (a release or the bump after it), run `pip install -e . --no-deps`
again: the installed metadata keeps the old version until then, and `tests/test_package.py` fails.

Use **only Git inside WSL** (VS Code: "WSL: Open Folder in WSL"). Mixing Windows Git and WSL Git
on the same OneDrive folder causes line-ending and permission noise.

## Daily workflow
```bash
git switch -c feat/<topic>
pytest -m "not slow"          # fast suite
pytest -m slow                # kill/resume tests
ruff check . && ruff format .
git push -u origin feat/<topic>   # from Windows Git Bash; the gh CLI is not installed
```
Commit messages: conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
The owner decides every merge; merges are made locally (`git merge --no-ff`) and `main` is pushed from Windows.

## OneDrive caveats
- OneDrive also syncs `.git/`. Push often; GitHub is the source of truth.
- Set the folder to "Always keep on this device".
- If OneDrive reports a sync conflict inside `.git/`, do not merge the copies. Re-clone from GitHub.

## Running gqteashmd unattended (workstation)
```bash
cd ~/gqteashmd-runs/my_molecule
nohup gqteashmd run input.toml > gqteashmd.out 2>&1 &     # or inside tmux
gqteashmd status input.toml
gqteashmd stop input.toml                              # graceful; `gqteashmd run` resumes
```

## Clean-room rule
Do not copy, translate or adapt code from SHARC 4.1. Implement from the literature.
SHARC may only be executed as a black box by scripts in `tools/sharc_compare/` (comparison
tests, marker `sharc`), never imported by `gqteashmd`.

## Making a release
(Release spec `docs/superpowers/specs/2026-11-25-release-0.1.0-design.md`, section 5.)

1. On the release branch: set `__version__` in `src/gqteashmd/__init__.py`, update
   `CHANGELOG.md`, and run `pip install -e . --no-deps`.
2. Gates on the branch head: CI green, and `pytest -W error` here with `SHARC_HOME`,
   `SHARC_PYTHON` and `ORCA_EXE` set.
3. Export the tested environment and commit it alone:
   `conda list -n gqteashmd-dev --explicit --md5 > environment.lock.txt`.
4. The owner merges into `main`; push `main`; CI runs on it.
5. On the merge commit: `tools/release/clean_install_check.sh VERSION` must print PASS
   (it builds the wheel from a clean clone, installs it into a fresh environment from the lock
   file, and runs `tools/release/water_check.toml` through the analysis; output under
   `~/release_check/VERSION/`).
6. `git tag -a vVERSION -m "gqteashmd VERSION"` on that commit; push the tag (owner's OK).
7. The owner creates the GitHub Release from the tag on github.com: the CHANGELOG entry as text,
   the wheel from `~/release_check/VERSION/dist/` and `environment.lock.txt` as assets. Attach
   nothing else (`report.json` and the project hold local paths and the host name).
8. Bump `__version__` to the next `.dev0` version, add an empty `Unreleased` section if needed,
   reinstall, commit and push.
