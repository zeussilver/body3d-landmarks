"""Scan quality check (SYNTHETIC): should the operator rescan before the subject steps off?

A scan is either ``ok`` or has one failure that needs a rescan. Failures are injected
into the simulated capture, so every label is known exactly:

- ``arms_down``: arms pressed against the torso (the protocol asks for arms held away);
- ``moved``: the subject turned and stepped during 2-3 consecutive views;
- ``cropped``: the subject stood off the platform centre, so part of the body leaves the
  field of view in some views;
- ``dropout``: a large region is missing in 1-2 views (a blocked camera or a lost frame),
  far beyond the scanner's normal small holes.

``ok`` scans use exactly the same noise and sway as the landmark experiments.

    python -m body3d.qa            # build data/qa/{dev,test} and results/scan_qa_examples.png
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from body3d import bodies, landmarks
from body3d.data import DATA, NOISE, RESULTS, SCANNER, SWAY, decode_depth, encode_depth
from body3d.scanner import BACKGROUND, Noise, add_noise, placement, rot_z, sway_views

CLASSES = ("ok", "arms_down", "moved", "cropped", "dropout")
QA_DIR = DATA / "qa"
SPLITS = {"dev": {"per_class": 100, "seed": 404}, "test": {"per_class": 50, "seed": 606}}

ARMS_DOWN_DEG = (38.0, 48.0)  # arm adduction; normal scans are within +-10 degrees
MOVE_TURN_DEG = (20.0, 35.0)  # extra turn during the moved views, either direction
MOVE_STEP_M = (0.05, 0.10)  # extra horizontal step during the moved views
MOVE_VIEWS = (2, 3)  # consecutive views affected
CROP_OFFSET_M = (0.45, 0.58)  # distance of the subject from the platform centre
CROP_MIN_EDGE_PX = 40  # body pixels in the outer 2 columns of the worst view, before sensor noise
CROP_STEP_M = 0.04
DROP_VIEWS = (1, 2)  # views affected
DROP_KINDS = ("blank", "band", "blob")


def assign(n_per_class: int, seed: int) -> tuple[bodies.Bodies, np.ndarray]:
    """Bodies (full adult range) and a balanced, shuffled class label per body.

    ``arms_down`` bodies are rebuilt with their arms adducted; everything else about the
    person (phenotype, local shape, stance) is kept.
    """
    n = n_per_class * len(CLASSES)
    b = bodies.sample_bodies(n, seed)
    rng = np.random.default_rng(seed)
    labels = rng.permutation(np.repeat(np.arange(len(CLASSES)), n_per_class))
    down = np.flatnonzero(labels == CLASSES.index("arms_down"))
    pose = b.pose[down].copy()
    pose[:, :2] = rng.uniform(*ARMS_DOWN_DEG, size=(len(down), 2))
    rebuilt = bodies.build(b.phenotype[down], b.local[down], pose)
    verts, pose_all = b.vertices.copy(), b.pose.copy()
    verts[down], pose_all[down] = rebuilt.vertices, pose
    return bodies.Bodies(verts, b.phenotype, b.local, pose_all, b.height, b.bmi), labels


def _edge_px(depth: np.ndarray) -> int:
    """Most body pixels in the two outer columns on either side, over views."""
    hit = np.isfinite(depth)
    return int(max(hit[k][:, :2].sum() + hit[k][:, -2:].sum() for k in range(len(hit))))


def _drop(depth: np.ndarray, rng: np.random.Generator) -> list[dict]:
    """Remove large regions from 1-2 random views in place; returns what was done."""
    n, h, w = depth.shape
    done = []
    for k in rng.choice(n, rng.integers(DROP_VIEWS[0], DROP_VIEWS[1] + 1), replace=False):
        kind = DROP_KINDS[rng.integers(len(DROP_KINDS))]
        hit = np.isfinite(depth[k])
        rows, cols = np.nonzero(hit)
        if kind == "blank":
            depth[k] = BACKGROUND
        elif kind == "band":  # a horizontal occluder across the body, 30-50% of its height
            top, bot = rows.min(), rows.max()
            span = int(rng.uniform(0.30, 0.50) * (bot - top))
            r0 = int(rng.integers(top, bot - span + 1))
            depth[k, r0 : r0 + span] = BACKGROUND
        else:  # a blob over the torso, 18-28 px radius (normal holes are 2-6 px)
            rad = rng.uniform(18, 28)
            cy = rng.uniform(np.percentile(rows, 30), np.percentile(rows, 60))
            cx = np.median(cols) + rng.uniform(-8, 8)
            rr, cc = np.mgrid[:h, :w]
            depth[k][(rr - cy) ** 2 + (cc - cx) ** 2 <= rad**2] = BACKGROUND
        done.append({"view": int(k), "kind": kind})
    return done


def qa_scan(v: np.ndarray, f: np.ndarray, lm: np.ndarray, label: str, rng: np.random.Generator,
            arm_w: np.ndarray, noise: Noise = NOISE) -> tuple[np.ndarray, np.ndarray, dict]:
    """One simulated capture with the failure ``label`` injected.

    Returns noisy depth (N, H, W) in metres, the landmark ground truth in the turntable
    frame (K, 3) and a dict describing the injected failure.
    """
    sc = SCANNER
    r, t = placement(rng, noise)
    info: dict = {}
    if label == "cropped":
        ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(*CROP_OFFSET_M)
        t = np.r_[dist * np.cos(ang), dist * np.sin(ang), 0.0]
        info = {"offset_m": round(float(dist), 3)}
    vb = sway_views(v, sc.n_views, rng, SWAY, arm_w)
    if label == "moved":
        m = int(rng.integers(MOVE_VIEWS[0], MOVE_VIEWS[1] + 1))
        k0 = int(rng.integers(sc.n_views))
        turn = rng.uniform(*MOVE_TURN_DEG) * rng.choice([-1.0, 1.0])
        ang, step = rng.uniform(0, 2 * np.pi), rng.uniform(*MOVE_STEP_M)
        shift = np.r_[step * np.cos(ang), step * np.sin(ang), 0.0]
        views = [(k0 + j) % sc.n_views for j in range(m)]
        for k in views:
            vb[k] = vb[k] @ rot_z(turn).T + shift
        info = {"views": views, "turn_deg": round(float(turn), 1), "step_m": round(float(step), 3)}
    depth, cos = sc.render(vb @ r.T + t, f)
    if label == "cropped":
        # The label means part of the body is really outside the image. If the offset points
        # where the body is narrow (its side faces the image edge), move the subject further out.
        while _edge_px(depth) < CROP_MIN_EDGE_PX:
            t = t * (1 + CROP_STEP_M / np.linalg.norm(t[:2]))
            depth, cos = sc.render(vb @ r.T + t, f)
        info = {"offset_m": round(float(np.linalg.norm(t[:2])), 3)}
    depth = add_noise(depth, cos, rng, noise)
    if label == "dropout":
        info = {"dropped": _drop(depth, rng)}
    return depth.astype(np.float32), (lm @ r.T + t).astype(np.float32), info


_worker: dict = {}


def _init(faces, idx, arm_w):
    _worker.update(faces=faces, idx=idx, arm_w=arm_w)


def _one(args):
    v, label, seed = args
    v = np.asarray(v, np.float64)
    w = _worker
    depth, lm, info = qa_scan(v, w["faces"], v[w["idx"]], CLASSES[label], np.random.default_rng(seed), w["arm_w"])
    return encode_depth(depth), lm, info


def build_split(name: str, workers: int | None = None) -> Path:
    """Render and cache a split under data/qa/<name>/ (re-created bit-for-bit from the seed)."""
    cfg = SPLITS[name]
    d = QA_DIR / name
    if (d / "meta.json").exists():
        return d
    b, labels = assign(cfg["per_class"], cfg["seed"])
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    jobs = [(v, int(l), [cfg["seed"], i]) for i, (v, l) in enumerate(zip(b.vertices, labels))]
    with ProcessPoolExecutor(workers, initializer=_init,
                             initargs=(bodies.faces(), landmarks.load(), bodies.arm_weights())) as ex:
        out = list(ex.map(_one, jobs, chunksize=8))
    depth, lm, infos = zip(*out)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "depth_mm.npy", np.stack(depth))
    np.save(d / "landmarks.npy", np.stack(lm))
    b.save(d / "bodies.npz")
    meta = {"data": "SYNTHETIC (Anny bodies, simulated turntable scans, injected failures)",
            "classes": list(CLASSES), "seed": cfg["seed"], "labels": [CLASSES[l] for l in labels],
            "failure": list(infos)}
    (d / "meta.json").write_text(json.dumps(meta) + "\n")
    return d


def load_split(name: str) -> dict:
    d = build_split(name)
    meta = json.loads((d / "meta.json").read_text())
    return {"depth_mm": np.load(d / "depth_mm.npy", mmap_mode="r"), "landmarks": np.load(d / "landmarks.npy"),
            "labels": np.array([CLASSES.index(l) for l in meta["labels"]]), "failure": meta["failure"],
            "bodies": bodies.Bodies.load(d / "bodies.npz")}


# ---------------------------------------------------------------- metrics


def metrics(y: np.ndarray, pred: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Multi-class and rescan-decision metrics. ``pred`` may contain -1 (no valid answer),
    which counts as wrong and as 'rescan' (the safe default)."""
    y, pred = np.asarray(y), np.asarray(pred)
    k = len(CLASSES)
    conf = np.zeros((k, k + 1), np.int64)  # last column: no valid answer
    np.add.at(conf, (y, np.where(pred < 0, k, pred)), 1)
    per = {}
    for c in range(k):
        tp, fn, fp = conf[c, c], conf[c].sum() - conf[c, c], conf[:, c].sum() - conf[c, c]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[CLASSES[c]] = {"recall": float(rec), "precision": float(prec),
                           "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0}
    correct = pred == y
    boot = np.random.default_rng(seed).integers(0, len(y), (n_boot, len(y)))
    bad, flag = y != 0, pred != 0  # a failure needs a rescan; an invalid answer is flagged
    return {
        "n": int(len(y)),
        "accuracy": float(correct.mean()),
        "accuracy_ci95": [float(v) for v in np.percentile(correct[boot].mean(1), [2.5, 97.5])],
        "macro_f1": float(np.mean([p["f1"] for p in per.values()])),
        "per_class": per,
        "rescan": {"failures_caught": float((flag & bad).sum() / bad.sum()),
                   "false_alarms_on_ok": float((flag & ~bad).sum() / (~bad).sum())},
        "invalid_answers": int((pred < 0).sum()),
        "confusion": {"rows_true_cols_pred": list(CLASSES) + ["invalid"], "matrix": conf.tolist()},
    }


def summary_line(m: dict) -> str:
    rec = " ".join(f"{c}={v['recall']:.2f}" for c, v in m["per_class"].items())
    return (f"acc {m['accuracy']:.3f} macroF1 {m['macro_f1']:.3f} | recall {rec} | "
            f"rescan: caught {m['rescan']['failures_caught']:.2f}, false alarms {m['rescan']['false_alarms_on_ok']:.2f}")


# ---------------------------------------------------------------- images for the models


def shade(depth: np.ndarray) -> np.ndarray:
    """One view, metres with inf background -> uint8 grey image: nearer is brighter, background black."""
    hit = np.isfinite(depth)
    out = np.zeros(depth.shape, np.uint8)
    if hit.any():
        d = depth[hit]
        lo, hi = np.percentile(d, 1), np.percentile(d, 99)
        out[hit] = (255 * (1.0 - 0.75 * np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1))).astype(np.uint8)
    return out


def grid_image(depth: np.ndarray, labels: bool = True):
    """(8, H, W) depth -> PIL image, views in a 2 x 4 grid: top row 0-135 deg, bottom row 180-315 deg,
    so each bottom view is the opposite of the view above it. Grey 2 px separators."""
    from PIL import Image, ImageDraw

    n, h, w = depth.shape
    cols, gap = n // 2, 2
    img = np.full((2 * h + gap, cols * w + (cols - 1) * gap), 96, np.uint8)
    for k in range(n):
        r, c = divmod(k, cols)
        img[r * (h + gap) : r * (h + gap) + h, c * (w + gap) : c * (w + gap) + w] = shade(depth[k])
    im = Image.fromarray(img).convert("RGB")
    if labels:
        draw = ImageDraw.Draw(im)
        for k in range(n):
            r, c = divmod(k, cols)
            draw.text((c * (w + gap) + 4, r * (h + gap) + 3), f"{SCANNER.yaws[k]:.0f} deg", fill=(255, 200, 0))
    return im


def view_images(depth: np.ndarray):
    """Each view padded to a square, for per-view embeddings."""
    from PIL import Image

    n, h, w = depth.shape
    out = []
    for k in range(n):
        sq = np.zeros((h, h), np.uint8)
        sq[:, (h - w) // 2 : (h - w) // 2 + w] = shade(depth[k])
        out.append(Image.fromarray(sq).convert("RGB"))
    return out


def image_path(split: str, i: int) -> Path:
    return QA_DIR / split / "img" / f"{i:04d}.png"


def write_images(split: str) -> None:
    s = load_split(split)
    (QA_DIR / split / "img").mkdir(exist_ok=True)
    for i in range(len(s["labels"])):
        p = image_path(split, i)
        if not p.exists():
            grid_image(decode_depth(np.asarray(s["depth_mm"][i]))).save(p, optimize=True)


def _describe(label: str, info: dict) -> str:
    """Class and failure record -> a short caption."""
    yaw = SCANNER.yaws
    if "views" in info:
        v = info["views"]
        return f"turned {abs(info['turn_deg']):.0f}° during {yaw[v[0]]:.0f}–{yaw[v[-1]]:.0f}°"
    if "offset_m" in info:
        return f"stood {info['offset_m'] * 100:.0f} cm off centre"
    if "dropped" in info:
        return ", ".join(f"{d['kind']} at {yaw[d['view']]:.0f}°" for d in info["dropped"])
    return "arms lowered onto the torso" if label == "arms_down" else "protocol pose, normal sensor noise"


def examples_figure(path: Path = RESULTS / "scan_qa_examples.png") -> None:
    """One dev example per class, as the models see it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = load_split("dev")
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(3.3 * len(CLASSES), 3.2))
    for ax, c in zip(axes, range(len(CLASSES))):
        i = int(np.flatnonzero(s["labels"] == c)[0])
        ax.imshow(grid_image(decode_depth(np.asarray(s["depth_mm"][i]))))
        ax.set_title(f"{CLASSES[c]}\n{_describe(CLASSES[c], s['failure'][i])}", fontsize=9)
        ax.axis("off")
    fig.suptitle("SYNTHETIC scans with injected failures (8 depth views each)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=100)


if __name__ == "__main__":
    for split in SPLITS:
        build_split(split)
        write_images(split)
        print(split, np.bincount(load_split(split)["labels"]).tolist())
    examples_figure()
