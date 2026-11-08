"""Rule-based scan check: geometry of the turntable, no learned model.

The key fact: the camera is orthographic, so the silhouette of view k and of the opposite
view k+4 (180 degrees later) are exact mirror images if the subject stood still. Every
rule compares what the scanner should see with what it saw:

- cropped: body pixels touch the image border;
- dropout: a view is nearly empty; or one view of an opposite pair is mostly contained in
  its partner's mirror but covers much less of it (occluded band, patch cut through the
  body); or a view has a hole, where the opposite view sees body, far larger than the
  sensor's normal holes (a hand on the hip encloses a hole in both views, so it is ignored);
- moved: both views of an opposite pair have large regions the other lacks;
- arms_down: no view shows daylight between the arms and the torso at chest-to-hip
  height (the protocol pose leaves a gap in the front and back views).

Thresholds are tuned on the dev split only (coordinate search on macro-F1).
"""

from __future__ import annotations

import json

import numpy as np
from scipy import ndimage

from body3d import qa
from body3d.data import RESULTS, decode_depth

FEATURES = ("border_px", "hole_px", "drop_asym", "move_sym", "arm_gap")
# Cascade order: the first rule that fires names the failure.
ORDER = (("cropped", "border_px", ">"), ("dropout", "hole_px", ">"), ("dropout", "drop_asym", ">"),
         ("moved", "move_sym", ">"), ("arms_down", "arm_gap", "<"))
ARM_BAND = (0.55, 0.72)  # fraction of standing height (from the feet) where the arm-torso gap is checked
BLANK_FRACTION = 0.25  # a view with less body area than this share of the median view is a lost frame


def _clean(mask: np.ndarray) -> np.ndarray:
    """Close the sensor's speckle (grazing-angle dropout, 1-px gaps)."""
    return ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=2, border_value=0)


def _runs(row: np.ndarray, min_len: int = 2) -> int:
    """Number of foreground runs of at least ``min_len`` pixels in a boolean row."""
    d = np.diff(np.r_[0, row.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int(((ends - starts) >= min_len).sum())


def features(depth: np.ndarray) -> dict[str, float]:
    """(8, H, W) depth in metres -> rule features for one scan."""
    raw = np.isfinite(depth)
    n, h, w = raw.shape
    m = np.stack([_clean(x) for x in raw])

    border = max(int(x[:, :2].sum() + x[:, -2:].sum() + x[:2].sum()) for x in raw)

    hole = 0
    area = m.reshape(n, -1).sum(1)
    # A near-empty view (a lost frame) is a dropout even when its partner is lost too.
    asym = 1.0 if area.min() < BLANK_FRACTION * np.median(area) else 0.0
    sym = 0.0
    for k in range(n):
        a, b = m[k], m[(k + n // 2) % n][:, ::-1]  # this view and the opposite view, mirrored
        # A hole only counts where the opposite view sees body: the gap a hand encloses
        # against the hip is in both views, a blocked patch of the sensor is in one.
        holes, nh = ndimage.label(ndimage.binary_fill_holes(a) & ~a & b)
        if nh:
            hole = max(hole, int(np.bincount(holes.ravel())[1:].max()))
        if b.sum() == 0:
            continue
        missing = (b & ~a).sum() / b.sum()  # what the partner sees and this view lacks
        extra = (a & ~b).sum() / max(a.sum(), 1)  # what this view sees and the partner lacks
        # dropout: this view is a cut-down copy of its partner (much missing, almost nothing extra);
        # a turned subject also loses area against its partner, but gains some elsewhere
        asym = max(asym, missing - 2 * extra)
        sym = max(sym, min(missing, extra))  # moved: both views disagree

    gap = 0.0
    for x in raw:
        rows = np.flatnonzero(x.any(1))
        if len(rows) < 20:
            continue
        top, bot = rows.min(), rows.max()  # image rows grow downwards: bot = feet
        lo, hi = bot - ARM_BAND[1] * (bot - top), bot - ARM_BAND[0] * (bot - top)
        band = x[int(lo) : int(hi) + 1]
        gap = max(gap, float(np.mean([_runs(r) >= 3 for r in band])))
    return {"border_px": border, "hole_px": hole, "drop_asym": asym, "move_sym": sym, "arm_gap": gap}


def feature_table(split: str) -> np.ndarray:
    s = qa.load_split(split)
    return np.array([[features(decode_depth(np.asarray(d)))[f] for f in FEATURES] for d in s["depth_mm"]])


def classify(x: np.ndarray, th: dict[int, float]) -> np.ndarray:
    """Feature rows (N, len(FEATURES)) + one threshold per cascade step -> class indices."""
    pred = np.zeros(len(x), np.int64)  # ok unless a rule fires
    decided = np.zeros(len(x), bool)
    for step, (cls, feat, op) in enumerate(ORDER):
        v = x[:, FEATURES.index(feat)]
        fire = (v > th[step]) if op == ">" else (v < th[step])
        new = fire & ~decided
        pred[new] = qa.CLASSES.index(cls)
        decided |= new
    return pred


def tune(x: np.ndarray, y: np.ndarray, passes: int = 4) -> dict[int, float]:
    """Coordinate search over candidate thresholds (feature quantiles), maximising macro-F1."""
    cands = {step: np.unique(np.quantile(x[:, FEATURES.index(feat)], np.linspace(0, 1, 201)))
             for step, (_, feat, _) in enumerate(ORDER)}
    th = {step: (c[-1] if ORDER[step][2] == ">" else c[0]) for step, c in cands.items()}  # all rules off
    for _ in range(passes):
        for step, c in cands.items():
            scores = []
            for t in c:
                trial = {**th, step: t}
                scores.append(qa.metrics(y, classify(x, trial))["macro_f1"])
            th[step] = float(c[int(np.argmax(scores))])
    return th


def main() -> None:
    import time

    xd, yd = feature_table("dev"), qa.load_split("dev")["labels"]
    th = tune(xd, yd)
    t0 = time.perf_counter()
    xt = feature_table("test")
    ms = (time.perf_counter() - t0) / len(xt) * 1e3
    yt = qa.load_split("test")["labels"]
    res = {"method": "rules (turntable geometry), thresholds tuned on dev",
           "thresholds": {f"{c}:{f}{op}": th[i] for i, (c, f, op) in enumerate(ORDER)},
           "dev": qa.metrics(yd, classify(xd, th)), "test": qa.metrics(yt, classify(xt, th)),
           "ms_per_scan_cpu": round(ms, 1),
           "feature_means_by_class_dev": {c: dict(zip(FEATURES, np.round(xd[yd == i].mean(0), 4).tolist()))
                                          for i, c in enumerate(qa.CLASSES)}}
    np.save(qa.QA_DIR / "test" / "pred_rules.npy", classify(xt, th))
    (RESULTS / "scan_qa_rules.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("thresholds", "ms_per_scan_cpu")}, indent=1))
    print("dev ", qa.summary_line(res["dev"]))
    print("test", qa.summary_line(res["test"]))


if __name__ == "__main__":
    main()
