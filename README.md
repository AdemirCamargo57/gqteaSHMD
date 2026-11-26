# gqteashmd

Automated surface-hopping molecular dynamics, from one input file:
geometry optimization → frequencies → Wigner sampling → initial excitations → trajectories →
populations and lifetimes. Electronic structure: PySCF. Linux only (Oracle Linux 9; WSL2 works).

**Website:** https://ademircamargo57.github.io/gqteaSHMD/

**Status:** latest release 0.1.0, the first release ([CHANGELOG.md](CHANGELOG.md)); `main` is the development snapshot after it. Singlet dynamics
with PySCF; triplets and spin-orbit coupling only on analytic and LVC models.

## Install a release
Download `environment.lock.txt` and the wheel `gqteashmd-0.1.0-py3-none-any.whl` from the
GitHub Release, then:

```bash
conda create -n gqteashmd-0.1.0 --file environment.lock.txt
conda activate gqteashmd-0.1.0
pip install --no-deps gqteashmd-0.1.0-py3-none-any.whl
gqteashmd --version     # gqteashmd 0.1.0 (commit ...)
```

For development (an editable install from a clone), see [docs/development.md](docs/development.md).

## Use
```bash
gqteashmd init input.toml      # write a commented template
gqteashmd validate input.toml
gqteashmd run input.toml       # run all stages; run again to resume
gqteashmd status input.toml
gqteashmd stop input.toml
```

Keep projects on a local Linux disk, never in OneDrive or on a Windows drive.

## Documents
- [USER_MANUAL.md](USER_MANUAL.md): input, commands, outputs, methods, validation and limitations.
- [CHANGELOG.md](CHANGELOG.md): what each release contains.

Licence: MIT.
