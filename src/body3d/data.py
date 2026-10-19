"""Fixed data splits and cached scans (SYNTHETIC).

The test set is 2,000 bodies, 400 per mesh-BMI band, sex-balanced, scanned once with
subject sway, sensor noise and a per-body seed. It is cached under ``data/`` (git-ignored) and
re-created bit-for-bit from the seeds if deleted.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from body3d import bodies, landmarks
from body3d.scanner import Noise, Scanner, Sway, scan

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

TEST_SEED = 3
TEST_PER_BAND = 400
TRAIN_SIZE = 8000
VAL_SIZE = 1000
NARROW_BMI = (18.5, 27.0)
# Training distributions: "full" = all plausible adults, "narrow" = mesh BMI 18.5-27 only.
DISTRIBUTIONS = {
    "full": {"bmi_range": None, "train_seed": 101, "val_seed": 201},
    "narrow": {"bmi_range": NARROW_BMI, "train_seed": 102, "val_seed": 202},
}
SCANNER = Scanner()
NOISE = Noise()
SWAY = Sway()

_worker: dict = {}


def _init_worker(faces, idx, arm_w, scanner, noise, sway):
    _worker.update(faces=faces, idx=idx, arm_w=arm_w, scanner=scanner, noise=noise, sway=sway)


def _scan_one(args):
    v, seed = args
    v = np.asarray(v, dtype=np.float64)
    w = _worker
    s = scan(v, w["faces"], v[w["idx"]], np.random.default_rng(seed), w["scanner"], w["noise"], w["sway"], w["arm_w"])
    return encode_depth(s["depth"]), s["landmarks"], s["visible"], s["observable"]


def encode_depth(depth: np.ndarray) -> np.ndarray:
    """Metres (inf = no return) -> uint16 millimetres (0 = no return). Lossless at 1 mm quantisation."""
    return np.where(np.isfinite(depth), np.rint(depth * 1e3), 0).astype(np.uint16)


def decode_depth(depth_mm: np.ndarray) -> np.ndarray:
    d = depth_mm.astype(np.float32) * 1e-3
    d[depth_mm == 0] = np.inf
    return d


def scan_many(vertices: np.ndarray, seeds: list, noise: Noise | None = NOISE, sway: Sway | None = SWAY,
              workers: int | None = None) -> dict[str, np.ndarray]:
    """Scan every body in parallel; seeds are anything ``np.random.default_rng`` accepts."""
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    init = (bodies.faces(), landmarks.load(), bodies.arm_weights(), SCANNER, noise, sway)
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=init) as ex:
        out = list(ex.map(_scan_one, zip(vertices, seeds), chunksize=8))
    depth, lm, vis, obs = (np.stack(x) for x in zip(*out))
    return {"depth_mm": depth, "landmarks": lm, "visible": vis, "observable": obs}


def _cached_split(name: str, make_bodies, seed: int) -> tuple[bodies.Bodies, dict[str, np.ndarray]]:
    d = DATA / name
    files = ("depth_mm", "landmarks", "visible", "observable")
    if not (d / "bodies.npz").exists() or not all((d / f"{f}.npy").exists() for f in files):
        d.mkdir(parents=True, exist_ok=True)
        b = make_bodies()
        scans = scan_many(b.vertices, [[seed, i] for i in range(len(b))])
        for f in files:
            np.save(d / f"{f}.npy", scans[f])
        b.save(d / "bodies.npz")
    return bodies.Bodies.load(d / "bodies.npz"), {f: np.load(d / f"{f}.npy", mmap_mode="r") for f in files}


def test_split() -> tuple[bodies.Bodies, dict[str, np.ndarray]]:
    """(bodies, scans) for the fixed 2,000-body test set; scans are memory-mapped."""
    return _cached_split("test", lambda: bodies.sample_stratified(TEST_PER_BAND, seed=TEST_SEED), TEST_SEED)


def val_split(dist: str) -> tuple[bodies.Bodies, dict[str, np.ndarray]]:
    """Fixed noisy scans for validation, drawn from the same distribution as training."""
    cfg = DISTRIBUTIONS[dist]
    return _cached_split(f"val_{dist}", lambda: bodies.sample_bodies(VAL_SIZE, cfg["val_seed"], cfg["bmi_range"]),
                         cfg["val_seed"])


def train_meshes(dist: str) -> Path:
    """Directory with ``vertices.npy`` (memory-mappable) and ``meta.npz`` for a training split.

    Training scans are rendered on the fly, so only meshes are stored.
    """
    cfg = DISTRIBUTIONS[dist]
    d = DATA / f"train_{dist}"
    if not (d / "meta.npz").exists():
        d.mkdir(parents=True, exist_ok=True)
        b = bodies.sample_bodies(TRAIN_SIZE, cfg["train_seed"], cfg["bmi_range"])
        np.save(d / "vertices.npy", b.vertices)
        np.savez(d / "meta.npz", **{f: getattr(b, f) for f in b.__dataclass_fields__ if f != "vertices"})
    return d


def train_meshes_extrapolated(dist: str = "narrow", seed: int = 303, iters: int = 3) -> Path:
    """E3: one adiposity-extrapolated variant per training subject, same index order.

    After pushing the controls, the height control is corrected by Newton steps so each
    variant keeps its subject's stature (Anny's weight control can shift it).
    """
    from body3d.augment import extrapolate

    d = DATA / f"train_{dist}_extrap"
    if (d / "meta.npz").exists():
        return d
    src = train_bodies(dist)
    ph, lc = extrapolate(np.asarray(src.phenotype), np.asarray(src.local), np.random.default_rng(seed))
    ph = bodies.keep_stature(ph, lc, src.pose, src.height, iters)
    b = bodies.build(ph, lc, src.pose)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "vertices.npy", b.vertices)
    np.savez(d / "meta.npz", **{f: getattr(b, f) for f in b.__dataclass_fields__ if f != "vertices"})
    return d


def train_bodies(dist: str) -> bodies.Bodies:
    d = train_meshes(dist)
    with np.load(d / "meta.npz") as m:
        return bodies.Bodies(np.load(d / "vertices.npy", mmap_mode="r"), *(m[f] for f in list(bodies.Bodies.__dataclass_fields__)[1:]))


if __name__ == "__main__":
    b, s = test_split()
    print("test:", bodies.describe(b))
    for dist in DISTRIBUTIONS:
        print(f"train_{dist}:", bodies.describe(train_bodies(dist)))
        print(f"val_{dist}:", bodies.describe(val_split(dist)[0]))
