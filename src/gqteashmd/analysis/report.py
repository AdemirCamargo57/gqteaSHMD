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


def _isc_lines(transitions: Any) -> list[str]:
    """The intersystem-crossing block: the transitions by kind (S-T and T-S first), the pairs
    and the first singlet-to-triplet time, censored like the S0 arrival."""
    if not isinstance(transitions, Mapping):
        return ["not computed"]
    out = ["Changes of the active spin-free state (complete trajectories, up to the horizon)"]
    kinds = transitions.get("kinds") or {}
    for kind in ("S-T", "T-S", "S-S", "T-T"):
        k = kinds.get(kind) or {}
        out.append(
            f"- {kind}: at hop {_n(k.get('at_hop'))}, without hop {_n(k.get('without_hop'))}"
        )
    out.append(f"- pairs: {_items(transitions.get('pairs'))}")
    # fixed order: analysis.json is written with sorted keys
    first = transitions.get("first_singlet_to_triplet_fs") or {}
    stats = {k: first.get(k) for k in ("n_crossed", "n_censored", "mean", "median")}
    out.append(f"- first S-to-T time (fs): {_items(stats)}")
    out.append(f"- first S-to-T time per trajectory (fs): {_items(first.get('per_trajectory'))}")
    return out


def analysis_markdown(summary: Mapping[str, Any]) -> str:
    s = summary
    out: list[str] = ["# Analysis", "", "## Settings"]
    settings = s.get("settings") or {}
    for key in (
        "estimator",
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
    out += [f"- n_complete: {_n(s.get('n_complete'))}"]
    out += [f"- n_censored: {_n(s.get('n_censored'))}"]
    censored_block = (s.get("estimators") or {}).get("censored") or {}
    out += [f"- empty_rows: {_n(censored_block.get('empty_rows'))}"]
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
    out += ["", "## Censored trajectories"]
    censored = s.get("censored") or []
    if censored:
        out += ["| task | outcome | reason | censored_at_fs |", "|---|---|---|---|"]
        for c in censored:
            out.append(
                f"| {c.get('task')} | {c.get('outcome')} | {c.get('reason')}"
                f" | {_n(c.get('censored_at_fs'))} |"
            )
    else:
        out.append("none censored")
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

    in_use = s.get("estimator")
    other = {"censored": "complete", "complete": "censored"}.get(str(in_use))
    other_fit = (((s.get("estimators") or {}).get(other) or {}).get("fit") or {}) if other else {}
    out += ["", "## Other estimator", f"- estimator: {_n(other)}"]
    for key in ("tau_fs", "tau_interval_fs", "t0_fs", "t0_interval_fs"):
        out.append(f"- {key}: {_n(other_fit.get(key))}")

    blocks = s.get("estimators") or {}
    out += ["", "## Lifetime range"]
    for name in ("complete", "censored", "hold_in_s0"):
        block = blocks.get(name)
        if not block:
            out.append(f"- {name}: -")
            continue
        f = block.get("fit") or {}
        out.append(
            f"- {name}: tau {_n(f.get('tau_fs'))} fs {_n(f.get('tau_interval_fs'))}, "
            f"t0 {_n(f.get('t0_fs'))} fs {_n(f.get('t0_interval_fs'))}"
        )
    r = s.get("tau_range_fs")
    if r:
        out.append(
            f"- range: {_n(r.get('min'))}-{_n(r.get('max'))} fs "
            f"({r.get('min_estimator')} to {r.get('max_estimator')})"
        )
    else:
        out.append("- range: -")
    out.append("Which estimate is closest to the truth is not known (USER_MANUAL 8.19).")

    hops = s.get("hops") or {}
    out += ["", "## Hops (complete trajectories)"]
    out.append(f"- pairs: {_items(hops.get('pairs'))}")
    out.append(f"- frustrated (by originating state): {_items(hops.get('frustrated_from'))}")
    out.append(f"- hops per trajectory: {_items(hops.get('per_trajectory'))}")
    out.append(f"- first hop (fs): {_items(hops.get('first_hop_fs'))}")
    out.append(f"- first S0 arrival (fs): {_items(hops.get('first_s0_arrival_fs'))}")
    if "transitions" in s:  # only a project with triplets has the key (spec 4.3)
        out += ["", "## Intersystem crossing", *_isc_lines(s.get("transitions"))]

    energy = s.get("energy") or {}
    out += ["", "## Energy (complete trajectories)"]
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
