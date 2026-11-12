"""JSON sanitising and the Markdown report of the analysis stage."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats by None and NumPy values by Python ones."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _n(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return format(x, ".4g") if math.isfinite(x) else "-"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_n(v) for v in x) + "]"
    return str(x)


def _items(d: Any) -> str:
    if not isinstance(d, Mapping) or not d:
        return "-"
    return ", ".join(f"{k} {_n(v)}" for k, v in d.items())


def analysis_markdown(summary: Mapping[str, Any]) -> str:
    s = summary
    out: list[str] = ["# Analysis", "", "## Settings"]
    settings = s.get("settings") or {}
    for key in (
        "fit_states",
        "fit_population",
        "n_bootstrap",
        "energy_jump_eV",
        "exclude_energy_jumps",
        "project_seed",
    ):
        out.append(f"- {key}: {_n(settings.get(key))}")
    out += ["", f"- n_selected: {_n(s.get('n_selected'))}"]
    out += [f"- n_included: {_n(s.get('n_included'))}"]
    out += [f"- small_ensemble: {_n(s.get('small_ensemble'))}"]
    out += [f"- horizon_fs: {_n(s.get('horizon_fs'))}", f"- dt_out_fs: {_n(s.get('dt_out_fs'))}"]
    out += ["", "## Outcomes"]
    for key, count in (s.get("outcomes") or {}).items():
        out.append(f"- {key}: {_n(count)}")

    out += ["", "## Excluded trajectories"]
    excluded = s.get("excluded") or []
    if excluded:
        out += ["| task | reason | t_final_fs | energy_jump_eV |", "|---|---|---|---|"]
        for e in excluded:
            out.append(
                f"| {e.get('task')} | {e.get('reason')} | {_n(e.get('t_final_fs'))}"
                f" | {_n(e.get('energy_jump_eV'))} |"
            )
    else:
        out.append("none excluded")
    cut = s.get("cut") or []
    out += ["", "## Cut at the horizon", ", ".join(cut) if cut else "none"]

    out += ["", "## Populations"]
    table = s.get("table") or []
    if table:
        labels = list((table[0].get("P") or {}).keys())
        head = ["time_fs"] + [f"P_{k}" for k in labels] + [f"rho_{k}" for k in labels]
        out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for row in table:
            cells = [_n(row.get("time_fs"))]
            cells += [_n((row.get("P") or {}).get(k)) for k in labels]
            cells += [_n((row.get("rho") or {}).get(k)) for k in labels]
            out.append("| " + " | ".join(cells) + " |")
    else:
        out.append("no table")

    fit = s.get("fit") or {}
    out += ["", "## Fit"]
    for key in (
        "states",
        "population",
        "A",
        "t0_fs",
        "tau_fs",
        "t0_interval_fs",
        "tau_interval_fs",
        "t0_std_fs",
        "tau_std_fs",
        "rms",
        "flags",
        "bootstrap_failures",
    ):
        out.append(f"- {key}: {_n(fit.get(key))}")

    hops = s.get("hops") or {}
    out += ["", "## Hops"]
    out.append(f"- pairs: {_items(hops.get('pairs'))}")
    out.append(f"- frustrated (by originating state): {_items(hops.get('frustrated_from'))}")
    out.append(f"- hops per trajectory: {_items(hops.get('per_trajectory'))}")
    out.append(f"- first hop (fs): {_items(hops.get('first_hop_fs'))}")
    out.append(f"- first S0 arrival (fs): {_items(hops.get('first_s0_arrival_fs'))}")

    energy = s.get("energy") or {}
    out += ["", "## Energy"]
    out.append(f"- drift_eV: {_items(energy.get('drift_eV'))}")
    out.append(f"- jump_threshold_eV: {_n(energy.get('jump_threshold_eV'))}")
    for f in energy.get("flagged") or []:
        out.append(f"- flagged {f.get('task')}: jump {_n(f.get('energy_jump_eV'))} eV")

    out += ["", "## Consistency (max |P - rho|)", _items(s.get("consistency"))]

    warnings = s.get("warnings") or []
    out += ["", "## Warnings"]
    out += [f"- {w}" for w in warnings] if warnings else ["none"]
    if s.get("stage_failed"):
        out += ["", f"**Stage failed: {s.get('error') or '-'}**"]
    out += [
        "",
        "## Files",
        "- populations.csv",
        "- populations.h5",
        "- hops.csv",
        "- analysis.json",
        "",
    ]
    return "\n".join(out)
