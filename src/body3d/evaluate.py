"""Landmark error metrics, stratified by body type, with bootstrap confidence intervals.

Every number in the README comes from a ``results/<run>.json`` written here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from body3d import bodies, landmarks

THRESHOLDS_MM = (10, 20)
IN_RANGE_BMI_BAND = 1  # 18.5-25: inside every training range, including the narrow one
N_BOOT = 1000


def errors_mm(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(N, K, 3) predictions and ground truth in metres -> (N, K) Euclidean error in mm."""
    return np.linalg.norm(np.asarray(pred, np.float64) - np.asarray(gt, np.float64), axis=-1) * 1e3


def _stats(err: np.ndarray, rng: np.random.Generator) -> dict:
    """err: (n, K) for one group of bodies."""
    per_body = err.mean(1)
    boot = per_body[rng.integers(0, len(per_body), (N_BOOT, len(per_body)))].mean(1)
    out = {
        "n": int(len(err)),
        "mean_mm": float(per_body.mean()),
        "ci95_mm": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "median_mm": float(np.median(err)),
    }
    out.update({f"pck{t}": float((err <= t).mean() * 100) for t in THRESHOLDS_MM})
    out["per_landmark_mean_mm"] = dict(zip(landmarks.NAMES, np.round(err.mean(0), 2).tolist()))
    return out


def strata(b: bodies.Bodies) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    return {
        "bmi": (bodies.BMI_LABELS, b.bmi_band),
        "sex": (bodies.SEX_LABELS, b.sex),
        "height": (bodies.HEIGHT_LABELS, b.height_band),
    }


def summarise(err: np.ndarray, b: bodies.Bodies, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {"overall": _stats(err, rng), "per_landmark": {}, "strata": {}}
    for j, name in enumerate(landmarks.NAMES):
        e = err[:, j]
        out["per_landmark"][name] = {"mean_mm": float(e.mean()), "median_mm": float(np.median(e)),
                                     **{f"pck{t}": float((e <= t).mean() * 100) for t in THRESHOLDS_MM}}
    worst = None
    for factor, (labels, band) in strata(b).items():
        groups = {lab: _stats(err[band == i], rng) for i, lab in enumerate(labels) if (band == i).any()}
        out["strata"][factor] = groups
        lab, g = max(groups.items(), key=lambda kv: kv[1]["mean_mm"])
        if worst is None or g["mean_mm"] > worst["mean_mm"]:
            worst = {"factor": factor, "band": lab, "mean_mm": g["mean_mm"]}
    bmi = out["strata"]["bmi"]
    in_range = bmi[bodies.BMI_LABELS[IN_RANGE_BMI_BAND]]["mean_mm"]
    w_lab, w = max(bmi.items(), key=lambda kv: kv[1]["mean_mm"])
    out["worst_bmi_band"] = {"band": w_lab, "mean_mm": w["mean_mm"], "in_range_mean_mm": in_range,
                             "gap_mm": w["mean_mm"] - in_range, "ratio": w["mean_mm"] / in_range}
    out["worst_band_any_factor"] = worst
    return out


def evaluate(run: str, pred: np.ndarray, gt: np.ndarray, b: bodies.Bodies, meta: dict | None = None,
             results_dir: Path | None = None) -> dict:
    """Score predictions and write ``results/<run>.json``."""
    from body3d.data import RESULTS

    res = {"run": run, "data": "SYNTHETIC (Anny bodies, simulated turntable scans)", **(meta or {}),
           **summarise(errors_mm(pred, gt), b)}
    path = (results_dir or RESULTS) / f"{run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")
    return res


def band_table(results: list[dict], factor: str = "bmi") -> str:
    """Markdown table: one row per run, mean error (mm) per band of ``factor``."""
    labels = list(results[0]["strata"][factor])
    lines = ["| run | overall | " + " | ".join(labels) + " | worst / in-range |",
             "|---|---|" + "---|" * len(labels) + "---|"]
    for r in results:
        cells = [f"{r['strata'][factor][l]['mean_mm']:.1f}" for l in labels]
        w = r["worst_bmi_band"]
        lines.append(f"| {r['run']} | {r['overall']['mean_mm']:.1f} | " + " | ".join(cells)
                     + f" | {w['ratio']:.2f}x ({w['band']}) |")
    return "\n".join(lines)


def load(run: str) -> dict:
    from body3d.data import RESULTS

    return json.loads((RESULTS / f"{run}.json").read_text())
