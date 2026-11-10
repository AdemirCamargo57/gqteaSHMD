"""Report-only FSSH-versus-exact ensembles on Tully's models (not a test; nothing is asserted).

These are the runs of the P7a-1 validation that are documented but deliberately not asserted
(USER_MANUAL.md section 8.14.1(a)); the asserted ones are in
``tests/validation/test_tully_exact.py``. The set-up (points, exact solver, FSSH settings,
the 4 SE + 0.02 rule) is shared through ``tests/validation/tully_common.py``.

- ``other``: every point with the *other* decoherence setting, n = 500 by default (SAC and DAC
  with EDC, Granucci-Persico convention by default; ECR without decoherence). EDC biases SAC
  at low momentum and DAC at high energy; ECR without decoherence misses the reflection.
- ``dac-low``: DAC ln E = -2 with the asserted setting (no decoherence), n = 1000, and the
  exact solver's convergence at that point. FSSH is known to deviate in this low-energy
  Stueckelberg region (Tully 1990); the point was classified report-only after its numbers
  were seen.

Every number is printed and written to one JSON file. Runs in WSL from the repository root,
for example (about 20-30 min with 8 workers on the development workstation):

    python tools/validation/tully_report.py --out ~/tully_report.json --workers 8

``--n-other`` and ``--n-dac-low`` lower the ensemble sizes (e.g. 4 for a smoke run);
``--only`` selects one of the two parts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "tests" / "validation" / "tully_common.py"


def load_common():
    name = "gqteashmd_validation_tully_common"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, COMMON)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _ensemble_record(common, point, result, seconds, settings):
    ref = common.exact(point)
    common.compare(point, ref, result, seconds, settings["tag"], asserted=False)
    return {
        "point": common.ids(point),
        "model": point.model,
        "k": point.k,
        "x0": point.x0,
        "x_exit": point.x_exit,
        "sigma": point.sigma,
        "settings": settings,
        "n_trajectories": result.n_trajectories,
        "n_finished": result.n_finished,
        "counts": result.counts(),
        "wall_s": seconds,
        "asserted_in_tests": False,
        "branches": common.comparison(point, ref, result),
    }


def other_decoherence(common, n: int, workers: int, edc_convention: str) -> list[dict]:
    records = []
    for point in common.REPORT_POINTS:
        other = "edc" if point.decoherence == "none" else "none"
        result, seconds = common.fssh(
            point, n=n, decoherence=other, workers=workers, edc_convention=edc_convention
        )
        tag = other if other == "none" else f"edc-{edc_convention}"
        settings = {"tag": tag, "hopping": "sharc", "decoherence": other}
        if other == "edc":
            settings["edc_convention"] = edc_convention
        records.append(_ensemble_record(common, point, result, seconds, settings))
    return records


def dac_low(common, n: int, workers: int) -> dict:
    point = common.DAC_LOW
    conv = common.convergence(point)
    wide = common.exact(point, 1, 90.0)
    result, seconds = common.fssh(point, n=n, decoherence=point.decoherence, workers=workers)
    settings = {"tag": "none", "hopping": "sharc", "decoherence": "none"}
    record = _ensemble_record(common, point, result, seconds, settings)
    record["exact_convergence"] = {
        "halved_dx_dt_max_change": conv["max_change"],
        "wider_grid_max_change": max(abs(wide[b] - conv["base"][b]) for b in common.BRANCHES),
        "norm_minus_1": conv["base"]["norm"] - 1.0,
        "t_end": conv["base"]["t_end"],
    }
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="JSON file to write")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-other", type=int, default=500)
    parser.add_argument("--n-dac-low", type=int, default=1000)
    parser.add_argument(
        "--edc-convention", default="granucci_persico", choices=("granucci_persico", "sharc")
    )
    parser.add_argument("--only", choices=("other", "dac-low"))
    args = parser.parse_args(argv)

    common = load_common()
    start = time.perf_counter()
    report: dict = {
        "schema_version": 1,
        "script": "tools/validation/tully_report.py",
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "seed": common.SEED,
        "rule": "|P_FSSH - P_exact| <= 4 sqrt(P_exact (1 - P_exact) / n_finished) + 0.02",
        "fssh_settings": "parallel_vel, frustrated keep, dt 0.5 au, 25 substeps, wigner",
        "workers": args.workers,
    }
    if args.only in (None, "dac-low"):
        report["dac_low_energy"] = dac_low(common, args.n_dac_low, args.workers)
    if args.only in (None, "other"):
        report["other_decoherence"] = other_decoherence(
            common, args.n_other, args.workers, args.edc_convention
        )
    report["wall_s"] = time.perf_counter() - start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport written to {args.out} ({report['wall_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
