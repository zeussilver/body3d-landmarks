"""Baseline B0: rigid template alignment + nearest scan point.

Fuse the turntable views into one point cloud, align the neutral template to it with
ICP restricted to rotation about z, translation and uniform scale, then snap each
template landmark to the nearest scan point. The template never changes shape, so
this baseline shows how much error comes from body-type variation alone.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

from body3d.scanner import Scanner, rot_z

N_TEMPLATE_POINTS = 4000
N_SCAN_POINTS = 4000


def similarity_yaw(p: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares s, R (rotation about z), t with s * R @ p + t ~ q, closed form."""
    pm, qm = p.mean(0), q.mean(0)
    pc, qc = p - pm, q - qm
    a = (pc[:, 0] * qc[:, 0] + pc[:, 1] * qc[:, 1]).sum()
    b = (pc[:, 0] * qc[:, 1] - pc[:, 1] * qc[:, 0]).sum()
    r = rot_z(np.degrees(np.arctan2(b, a)))
    s = float((qc * (pc @ r.T)).sum() / (pc**2).sum())
    return s, r, qm - s * r @ pm


def apply(s: float, r: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return s * p @ r.T + t


def _band_median_xy(p: np.ndarray, top: float) -> np.ndarray:
    band = p[(p[:, 2] > 0.5 * top) & (p[:, 2] < 0.7 * top)]
    return np.r_[np.median(band[:, :2], axis=0), 0.0]


def icp_similarity(tpl: np.ndarray, scan: np.ndarray, iters: int = 30, trim: float = 0.9, seed: int = 0):
    """Align template points (M, 3) to scan points (S, 3); returns (s, R, t).

    Correspondences go both ways (template -> scan and scan -> template), and the
    worst (1 - trim) of each direction are dropped to cope with holes and pose jitter.
    """
    scan_top, tpl_top = scan[:, 2].max(), tpl[:, 2].max()
    s = scan_top / tpl_top
    r = np.eye(3)
    t = _band_median_xy(scan, scan_top) - s * _band_median_xy(tpl, tpl_top)
    tree = cKDTree(scan)
    rng = np.random.default_rng(seed)
    sub = scan[rng.choice(len(scan), min(N_SCAN_POINTS, len(scan)), replace=False)]
    for _ in range(iters):
        moved = apply(s, r, t, tpl)
        d1, j1 = tree.query(moved)
        d2, j2 = cKDTree(moved).query(sub)
        k1, k2 = d1 <= np.quantile(d1, trim), d2 <= np.quantile(d2, trim)
        s, r, t = similarity_yaw(np.r_[tpl[k1], tpl[j2[k2]]], np.r_[scan[j1[k1]], sub[k2]])
    return s, r, t


def template_points(vertices: np.ndarray, faces: np.ndarray, seed: int = 0) -> np.ndarray:
    """Area-weighted fixed subset of template vertex indices (dense hands/face would dominate otherwise)."""
    tri = vertices[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    w = np.zeros(len(vertices))
    np.add.at(w, faces.ravel(), np.repeat(area / 3, 3))
    return np.random.default_rng(seed).choice(len(vertices), N_TEMPLATE_POINTS, replace=False, p=w / w.sum())


def predict(depth: np.ndarray, tpl_pts: np.ndarray, tpl_lm: np.ndarray, scanner: Scanner = Scanner()):
    """Depth (N_views, H, W) in metres -> (snapped (K, 3), unsnapped (K, 3)) landmark estimates."""
    scan = np.concatenate(scanner.backproject(depth))
    s, r, t = icp_similarity(tpl_pts, scan)
    lm = apply(s, r, t, tpl_lm)
    snapped = scan[cKDTree(scan).query(lm)[1]]
    return snapped, lm


_worker: dict = {}


def _init(tpl_pts, tpl_lm):
    _worker.update(tpl_pts=tpl_pts, tpl_lm=tpl_lm)


def _predict_one(depth_mm):
    from body3d.data import decode_depth

    return predict(decode_depth(np.asarray(depth_mm)), _worker["tpl_pts"], _worker["tpl_lm"])


def run_b0(depth_mm: np.ndarray, workers: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """B0 on a stack of encoded scans (N, views, H, W) -> (snapped, unsnapped) (N, K, 3)."""
    from body3d import bodies, landmarks

    t = bodies.template().vertices[0]
    tpl_pts = t[template_points(t, bodies.faces())]
    tpl_lm = t[landmarks.load()]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(tpl_pts, tpl_lm)) as ex:
        out = list(ex.map(_predict_one, depth_mm, chunksize=8))
    snapped, raw = (np.stack(x).astype(np.float32) for x in zip(*out))
    return snapped, raw


def main() -> None:
    import time

    from body3d import data, evaluate

    b, scans = data.test_split()
    t0 = time.perf_counter()
    snapped, raw = run_b0(scans["depth_mm"])
    secs = time.perf_counter() - t0
    np.save(data.DATA / "test" / "pred_b0.npy", snapped)
    meta = {"method": "B0: ICP (yaw + translation + uniform scale) of the neutral template, then nearest scan point",
            "seconds_total": round(secs, 1)}
    res = evaluate.evaluate("b0", snapped, scans["landmarks"], b, meta)
    evaluate.evaluate("b0_nosnap", raw, scans["landmarks"], b, {**meta, "method": meta["method"].replace(", then nearest scan point", ", no snapping")})
    print(evaluate.band_table([res, evaluate.load("b0_nosnap")]))
    print("worst any factor:", res["worst_band_any_factor"])


if __name__ == "__main__":
    main()
