"""Output writers of the analysis stage (populations.h5/.csv, hops.csv)."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from gqteashmd import __version__
from gqteashmd.analysis.hops import HopEvent


def write_populations_h5(
    path: Path,
    *,
    time_fs: ArrayLike,
    labels: Sequence[str],
    classical: ArrayLike,
    classical_lower: ArrayLike,
    classical_upper: ArrayLike,
    quantum: ArrayLike,
    quantum_lower: ArrayLike,
    quantum_upper: ArrayLike,
    fit_target: ArrayLike,
    fit_curve: ArrayLike,
    attrs: Mapping[str, Any],
) -> None:
    import h5py

    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["schema_version"] = 1
            f.attrs["gqteashmd_version"] = __version__
            for key, value in attrs.items():
                f.attrs[key] = value
            names = np.array(list(labels), dtype=h5py.string_dtype())
            f.create_dataset("state_labels", data=names).attrs["units"] = "-"
            for name, data, units in (
                ("time_fs", time_fs, "fs"),
                ("classical", classical, "-"),
                ("classical_lower", classical_lower, "-"),
                ("classical_upper", classical_upper, "-"),
                ("quantum", quantum, "-"),
                ("quantum_lower", quantum_lower, "-"),
                ("quantum_upper", quantum_upper, "-"),
                ("fit_target", fit_target, "-"),
                ("fit_curve", fit_curve, "-"),
            ):
                f.create_dataset(name, data=np.asarray(data, dtype=float)).attrs["units"] = units
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _fmt(x: float) -> str:
    return format(float(x), ".6e") if np.isfinite(x) else ""


def format_populations_csv(
    *,
    time_fs: ArrayLike,
    labels: Sequence[str],
    classical: ArrayLike,
    classical_lower: ArrayLike,
    classical_upper: ArrayLike,
    quantum: ArrayLike,
    quantum_lower: ArrayLike,
    quantum_upper: ArrayLike,
    fit_target: ArrayLike,
    fit_curve: ArrayLike,
    comments: Sequence[str],
) -> str:
    t = np.asarray(time_fs, dtype=float)
    blocks = [
        ("P", classical, classical_lower, classical_upper),
        ("rho", quantum, quantum_lower, quantum_upper),
    ]
    arrays = [
        (np.asarray(m, float), np.asarray(lo, float), np.asarray(hi, float))
        for _, m, lo, hi in blocks
    ]
    header = ["time_fs"]
    for prefix, *_ in blocks:
        for lab in labels:
            header += [f"{prefix}_{lab}", f"{prefix}_{lab}_lower", f"{prefix}_{lab}_upper"]
    header += ["fit_target", "fit_curve"]
    target = np.asarray(fit_target, dtype=float)
    curve = np.asarray(fit_curve, dtype=float)
    lines = [f"# {c}" for c in comments] + [",".join(header)]
    for i in range(t.size):
        row = [_fmt(t[i])]
        for mid, lo, hi in arrays:
            for k in range(len(labels)):
                row += [_fmt(mid[i, k]), _fmt(lo[i, k]), _fmt(hi[i, k])]
        row += [_fmt(target[i]), _fmt(curve[i])]
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def format_hops_csv(events: Sequence[HopEvent], labels: Sequence[str]) -> str:
    lines = ["task,time_fs,from,to,frustrated,included,after_horizon"]
    for e in events:
        to = "" if e.to_state is None else labels[e.to_state]
        lines.append(
            f"{e.task},{e.time_fs:.6f},{labels[e.from_state]},{to},"
            f"{int(e.frustrated)},{int(e.included)},{int(e.after_horizon)}"
        )
    return "\n".join(lines) + "\n"
