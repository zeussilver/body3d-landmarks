"""Aggregate ``results/<cond>_s<seed>.json`` into ``results/summary.json`` and README tables.

Gap closure on the narrow model's worst BMI band (paired by seed):
    closure = (E1 - Ex) / (E1 - E0)
"""

from __future__ import annotations

import json
import re

import numpy as np

from body3d import bodies
from body3d.data import RESULTS

CONDS = {
    "b0": "B0 rigid template + nearest point",
    "e0": "E0 trained on full range",
    "e1": "E1 trained on BMI 18.5-27",
    "e2": "E2 = E1 + normal-offset inflation",
    "e3": "E3 = E1 + shape-space extrapolation",
}
IN_RANGE = bodies.BMI_LABELS[1]


def collect(results_dir=RESULTS) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {c: {} for c in CONDS}
    for path in results_dir.glob("*.json"):
        m = re.fullmatch(r"(e\d)_s(\d+)", path.stem)
        if m and m[1] in runs:
            runs[m[1]][int(m[2])] = json.loads(path.read_text())
        elif path.stem == "b0":
            runs["b0"][0] = json.loads(path.read_text())
    return {c: r for c, r in runs.items() if r}


def _band(r: dict, band: str) -> float:
    return r["strata"]["bmi"][band]["mean_mm"]


def _ms(values) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def summarise(runs: dict[str, dict[int, dict]]) -> dict:
    out = {"data": "SYNTHETIC", "conditions": {}}
    for c, by_seed in runs.items():
        res = list(by_seed.values())
        out["conditions"][c] = {
            "label": CONDS[c], "seeds": sorted(by_seed),
            "overall_mm": _ms([r["overall"]["mean_mm"] for r in res]),
            "pck10": _ms([r["overall"]["pck10"] for r in res]),
            "pck20": _ms([r["overall"]["pck20"] for r in res]),
            "bmi_mm": {b: _ms([_band(r, b) for r in res]) for b in bodies.BMI_LABELS},
        }
    if {"e0", "e1"} <= runs.keys():
        worst = max(bodies.BMI_LABELS, key=lambda b: np.mean([_band(r, b) for r in runs["e1"].values()]))
        out["e1_worst_band"] = worst
        for c in ("e2", "e3"):
            seeds = sorted(set(runs["e0"]) & set(runs["e1"]) & set(runs.get(c, {})))
            if not seeds:
                continue
            clo = [(_band(runs["e1"][s], worst) - _band(runs[c][s], worst))
                   / (_band(runs["e1"][s], worst) - _band(runs["e0"][s], worst)) * 100 for s in seeds]
            dr = [_band(runs[c][s], IN_RANGE) - _band(runs["e1"][s], IN_RANGE) for s in seeds]
            out[f"{c}_vs_e1"] = {
                "paired_seeds": seeds,
                "gap_closure_pct": {"mean": float(np.mean(clo)), "std": float(np.std(clo)), "per_seed": clo},
                "in_range_change_mm": {"mean": float(np.mean(dr)), "std": float(np.std(dr)), "per_seed": dr},
                "A4_met": bool(np.mean(clo) >= 50 and np.mean(dr) <= 1.0),
            }
    return out


def _pm(x: dict, n: int) -> str:
    return f"{x['mean']:.2f}" if n == 1 else f"{x['mean']:.2f} ± {x['std']:.2f}"


def table(summary: dict) -> str:
    lines = ["| model | seeds | overall | " + " | ".join(bodies.BMI_LABELS) + " |",
             "|---|---|---|" + "---|" * len(bodies.BMI_LABELS)]
    for c, row in summary["conditions"].items():
        n = len(row["seeds"])
        lines.append(f"| {row['label']} | {n} | {_pm(row['overall_mm'], n)} | "
                     + " | ".join(_pm(row["bmi_mm"][b], n) for b in bodies.BMI_LABELS) + " |")
    for c in ("e2", "e3"):
        if f"{c}_vs_e1" in summary:
            s = summary[f"{c}_vs_e1"]
            lines.append(f"\n{c.upper()} closes {s['gap_closure_pct']['mean']:.0f}% ± {s['gap_closure_pct']['std']:.0f}% of the "
                         f"E1→E0 gap on the {summary['e1_worst_band']} band; in-range change "
                         f"{s['in_range_change_mm']['mean']:+.2f} mm (A4 {'met' if s['A4_met'] else 'not met'}).")
    return "\n".join(lines)


def main() -> None:
    summary = summarise(collect())
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(table(summary))


if __name__ == "__main__":
    main()
