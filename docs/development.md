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
Use **only Git inside WSL** (VS Code: "WSL: Open Folder in WSL"). Mixing Windows Git and WSL Git
on the same OneDrive folder causes line-ending and permission noise.

## Daily workflow
```bash
git switch -c feat/<topic>
pytest -m "not slow"          # fast suite
pytest -m slow                # kill/resume tests
ruff check . && ruff format .
git push -u origin feat/<topic> && gh pr create
```
Commit messages: conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
`main` is protected: PR + green CI + one review.

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
