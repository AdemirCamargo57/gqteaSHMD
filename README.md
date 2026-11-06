# gqteashmd

Automated surface-hopping molecular dynamics workflow:
geometry optimization → frequencies → Wigner sampling → initial conditions → trajectories,
driven from a single input file. Electronic structure: PySCF.

Status: early development (P1: foundation & orchestrator). Linux only.

```bash
gqteashmd init input.toml      # write a commented template
gqteashmd validate input.toml
gqteashmd run input.toml       # run all stages; re-run to resume
gqteashmd status input.toml
gqteashmd stop input.toml
```

Developer setup: see [docs/development.md](docs/development.md).
Licence: MIT.
