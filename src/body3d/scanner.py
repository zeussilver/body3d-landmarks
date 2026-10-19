"""Simulated turntable depth scanner (SYNTHETIC; a simplified stand-in, not modelled on any real device).

The subject stands on a turntable that is captured at ``n_views`` known yaw angles by
a fixed orthographic depth camera in front of it. Frames:

- body frame: from ``bodies`` (metres, z up, faces -y, pelvis at x = y = 0);
- turntable frame: the body after a random facing angle and centre offset
  (how a person actually stands on the platform); ground truth lives here;
- view k: the turntable frame rotated by yaw_k about z. Image column follows x,
  image row follows -z, depth is y + ``distance`` (smaller = closer to the camera).

Rasteriser: edge-function test at every pixel centre inside each triangle's bounding
box, barycentric depth interpolation (exact under orthographic projection) and a
z-buffer. Plain numpy, deterministic, and straightforward to port.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BACKGROUND = np.inf


def rasterise(col: np.ndarray, row: np.ndarray, d: np.ndarray, f: np.ndarray, h: int, w: int):
    """Z-buffer one view. Vertex pixel coordinates (V,) where integers are pixel centres,
    vertex depth (V,), faces (T, 3). Returns flat (h*w,) depth and the winning face per pixel."""
    tc, tr, td = col[f], row[f], d[f]
    c0 = np.clip(np.ceil(tc.min(1)), 0, w).astype(np.int64)
    c1 = np.clip(np.floor(tc.max(1)), -1, w - 1).astype(np.int64)
    r0 = np.clip(np.ceil(tr.min(1)), 0, h).astype(np.int64)
    r1 = np.clip(np.floor(tr.max(1)), -1, h - 1).astype(np.int64)
    nc, nr = np.clip(c1 - c0 + 1, 0, None), np.clip(r1 - r0 + 1, 0, None)
    cnt = nc * nr
    tid = np.repeat(np.arange(len(f)), cnt)  # one entry per candidate pixel centre in a face bbox
    local = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    pc, pr = c0[tid] + local % nc[tid], r0[tid] + local // nc[tid]
    x, y, z = tc[tid], tr[tid], td[tid]
    den = (y[:, 1] - y[:, 2]) * (x[:, 0] - x[:, 2]) + (x[:, 2] - x[:, 1]) * (y[:, 0] - y[:, 2])
    ok = np.abs(den) > 1e-12
    den = np.where(ok, den, 1.0)
    w0 = ((y[:, 1] - y[:, 2]) * (pc - x[:, 2]) + (x[:, 2] - x[:, 1]) * (pr - y[:, 2])) / den
    w1 = ((y[:, 2] - y[:, 0]) * (pc - x[:, 2]) + (x[:, 0] - x[:, 2]) * (pr - y[:, 2])) / den
    w2 = 1.0 - w0 - w1
    ok &= (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
    flat = (pr * w + pc)[ok]
    dep = (w0 * z[:, 0] + w1 * z[:, 1] + w2 * z[:, 2])[ok]
    tid = tid[ok]
    zbuf = np.full(h * w, BACKGROUND)
    np.minimum.at(zbuf, flat, dep)
    face = np.full(h * w, -1, dtype=np.int64)
    win = dep <= zbuf[flat]
    face[flat[win]] = tid[win]
    return zbuf, face


def rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


@dataclass(frozen=True)
class Scanner:
    n_views: int = 8
    height: int = 256  # rows
    width: int = 160  # columns
    pixel: float = 0.0085  # metres per pixel -> 2.18 m x 1.36 m field of view
    distance: float = 1.2  # nominal camera-to-axis distance, only offsets depth values

    @property
    def yaws(self) -> np.ndarray:
        return np.arange(self.n_views) * 360.0 / self.n_views

    def view_points(self, p: np.ndarray, k: int) -> np.ndarray:
        """Turntable-frame points (..., 3) -> view-k frame."""
        return p @ rot_z(self.yaws[k]).T

    def to_pixels(self, pv: np.ndarray):
        """View-frame points -> continuous (col, row) where integers are pixel centres, and depth."""
        col = (pv[..., 0] + self.width * self.pixel / 2) / self.pixel - 0.5
        row = (self.height * self.pixel - pv[..., 2]) / self.pixel - 0.5
        return col, row, pv[..., 1] + self.distance

    def pixel_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """(H, W) view-frame x and z of every pixel centre."""
        c = (np.arange(self.width) + 0.5) * self.pixel - self.width * self.pixel / 2
        r = self.height * self.pixel - (np.arange(self.height) + 0.5) * self.pixel
        return np.broadcast_to(c, (self.height, self.width)), np.broadcast_to(r[:, None], (self.height, self.width))

    def render(self, v: np.ndarray, f: np.ndarray):
        """Clean depth (N, H, W) with BACKGROUND where nothing is hit, and |cos| between the
        surface normal and the viewing ray (N, H, W). ``v`` is (V, 3), or (N, V, 3) when
        the subject moves between views."""
        shape = (self.n_views, self.height, self.width)
        depth, cos = np.empty(shape), np.zeros(shape)
        for k in range(self.n_views):
            vk = v[k] if v.ndim == 3 else v
            if k == 0 or v.ndim == 3:
                tri = vk[f]
                nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            r = rot_z(self.yaws[k])
            col, row, d = self.to_pixels(vk @ r.T)
            zbuf, face = rasterise(col, row, d, f, self.height, self.width)
            depth[k] = zbuf.reshape(self.height, self.width)
            hit = face >= 0
            c = np.zeros(self.height * self.width)
            c[hit] = np.abs(nrm[face[hit]] @ r[1])  # view direction is +y in the view frame
            cos[k] = c.reshape(self.height, self.width)
        return depth, cos

    def backproject(self, depth: np.ndarray) -> list[np.ndarray]:
        """Per view, (M, 3) turntable-frame points for valid pixels."""
        xc, zc = self.pixel_centres()
        out = []
        for k in range(self.n_views):
            m = np.isfinite(depth[k])
            pv = np.stack([xc[m], depth[k][m] - self.distance, zc[m]], 1)
            out.append(pv @ rot_z(self.yaws[k]))  # inverse rotation
        return out

    def project(self, p: np.ndarray):
        """Turntable points (K, 3) -> per view (N, K) continuous col, row and depth."""
        cols, rows, ds = zip(*(self.to_pixels(self.view_points(p, k)) for k in range(self.n_views)))
        return np.stack(cols), np.stack(rows), np.stack(ds)

    def visibility(self, p: np.ndarray, depth: np.ndarray, tol: float = 0.01) -> np.ndarray:
        """(N, K) True where point k is the visible surface in view n (within ``tol`` metres)."""
        col, row, d = self.project(p)
        ci = np.clip(np.rint(col).astype(int), 0, self.width - 1)
        ri = np.clip(np.rint(row).astype(int), 0, self.height - 1)
        inside = (np.rint(col) == ci) & (np.rint(row) == ri)
        z = depth[np.arange(self.n_views)[:, None], ri, ci]
        return inside & (np.abs(d - z) < tol)

    def observable(self, p: np.ndarray, depth: np.ndarray, tol: float = 0.01) -> np.ndarray:
        """(N, K) visible, or on the silhouette (within 1 px of background).

        Horizontal cameras see upward/downward-facing points (top of head, crotch) only
        as silhouette extremes; their position is still recoverable from the outline.
        """
        from scipy.ndimage import binary_dilation, binary_erosion

        col, row, _ = self.project(p)
        ci = np.clip(np.rint(col).astype(int), 0, self.width - 1)
        ri = np.clip(np.rint(row).astype(int), 0, self.height - 1)
        hit = np.isfinite(depth)
        edge = np.stack([binary_dilation(h) & ~binary_erosion(h) for h in hit])
        return self.visibility(p, depth, tol) | edge[np.arange(self.n_views)[:, None], ri, ci]


@dataclass(frozen=True)
class Noise:
    """Depth-sensor imperfections. Values are plausible guesses, not calibrated to any device."""

    sigma_mm: tuple[float, float] = (1.0, 3.0)  # per-scan base noise, grows at grazing angles
    grazing_cos: tuple[float, float] = (0.10, 0.25)  # dropout prob ramps from 1 to 0 across this
    holes: tuple[int, int] = (0, 4)  # random blob holes per view
    hole_radius_px: tuple[float, float] = (2.0, 6.0)
    quant_mm: float = 1.0
    facing_deg: float = 15.0  # subject facing on the platform, +-
    offset_m: float = 0.02  # subject centre offset on the platform, +-


@dataclass(frozen=True)
class Sway:
    """Subject motion during the ~60 s rotation. Views are ~7.5 s apart, longer than a
    postural sway cycle, so each view gets an independent perturbation. Standard
    deviations below are guesses, not measurements. Per-scan means are removed, so the
    ground truth is the subject's average position."""

    drift_mm: float = 3.0  # horizontal translation, per axis
    lean_deg: float = 0.25  # rotation about x and y, pivot on the floor (head moves ~7 mm)
    twist_deg: float = 0.4  # rotation about z
    arm_deg: float = 1.5  # arm abduction, each side


def arm_pivots(v: np.ndarray, arm_w: np.ndarray) -> np.ndarray:
    """(2, 3) left/right shoulder pivots: centroid of the vertices where the arm skinning
    weight is about one half (the torso/arm boundary)."""
    return np.stack([v[(arm_w[:, s] > 0.3) & (arm_w[:, s] < 0.7)].mean(0) for s in range(2)])


def sway_views(v: np.ndarray, n_views: int, rng: np.random.Generator, sway: Sway = Sway(),
               arm_w: np.ndarray | None = None) -> np.ndarray:
    """Body-frame vertices (V, 3) -> (n_views, V, 3), one independently perturbed copy per view.

    Arm drift blends each vertex between rest and rotated-about-the-shoulder by its arm
    skinning weight (linear blend skinning), so the shoulder stays smooth.
    """
    def draw(sd, *size):
        x = rng.normal(0.0, sd, (n_views, *size))
        return x - x.mean(0)

    drift, lean = draw(sway.drift_mm * 1e-3, 2), draw(sway.lean_deg, 2)
    twist, arm = draw(sway.twist_deg), draw(sway.arm_deg, 2)
    piv = arm_pivots(v, arm_w) if arm_w is not None else None
    out = np.empty((n_views, *v.shape))
    for k in range(n_views):
        vk = v
        if arm_w is not None:
            for side, sign in ((0, 1.0), (1, -1.0)):  # +y rotation brings the left (+x) arm inward
                r, p, w = rot_y(sign * arm[k, side]), piv[side], arm_w[:, side : side + 1]
                vk = vk + w * ((vk - p) @ r.T + p - vk)
        r = rot_z(twist[k]) @ rot_y(lean[k, 1]) @ rot_x(lean[k, 0])
        out[k] = vk @ r.T + np.r_[drift[k], 0.0]
    return out


def placement(rng: np.random.Generator, noise: Noise = Noise()) -> tuple[np.ndarray, np.ndarray]:
    """Random (3x3 rotation, 3-vector offset) placing a body on the turntable."""
    r = rot_z(rng.uniform(-noise.facing_deg, noise.facing_deg))
    t = np.r_[rng.uniform(-noise.offset_m, noise.offset_m, 2), 0.0]
    return r, t


def add_noise(depth: np.ndarray, cos: np.ndarray, rng: np.random.Generator, noise: Noise = Noise()) -> np.ndarray:
    """Noisy copy of a clean render: grazing-angle dropout, blob holes, Gaussian noise, quantisation."""
    out = depth.copy()
    hit = np.isfinite(out)
    lo, hi = noise.grazing_cos
    p_drop = np.clip((hi - cos) / (hi - lo), 0.0, 1.0)
    out[hit & (rng.random(out.shape) < p_drop)] = BACKGROUND
    n, h, w = out.shape
    rr, cc = np.mgrid[:h, :w]
    for k in range(n):
        ys, xs = np.nonzero(np.isfinite(out[k]))
        for _ in range(rng.integers(noise.holes[0], noise.holes[1] + 1)):
            if len(ys) == 0:
                break
            j = rng.integers(len(ys))
            rad = rng.uniform(*noise.hole_radius_px)
            out[k][(rr - ys[j]) ** 2 + (cc - xs[j]) ** 2 <= rad**2] = BACKGROUND
    hit = np.isfinite(out)
    sigma = rng.uniform(*noise.sigma_mm) * 1e-3 / np.maximum(cos[hit], 0.3)
    out[hit] += rng.normal(0.0, 1.0, hit.sum()) * sigma
    q = noise.quant_mm * 1e-3
    out[hit] = np.round(out[hit] / q) * q
    return out


def scan(v: np.ndarray, f: np.ndarray, lm: np.ndarray, rng: np.random.Generator,
         scanner: Scanner = Scanner(), noise: Noise | None = Noise(), sway: Sway | None = None,
         arm_w: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Place a body on the turntable, render all views, optionally with subject sway
    between views and sensor noise.

    Returns depth (N, H, W), visibility and observability of each landmark (N, K) and the
    landmark ground truth (the average position) in the turntable frame (K, 3).
    """
    r, t = placement(rng, noise or Noise())
    lm_t = lm @ r.T + t
    vb = sway_views(v, scanner.n_views, rng, sway, arm_w) if sway is not None else v
    depth, cos = scanner.render(vb @ r.T + t, f)
    vis, obs = scanner.visibility(lm_t, depth), scanner.observable(lm_t, depth)
    if noise is not None:
        depth = add_noise(depth, cos, rng, noise)
    return {"depth": depth.astype(np.float32), "visible": vis, "observable": obs, "landmarks": lm_t.astype(np.float32)}


def ray_cast_depth(v: np.ndarray, f: np.ndarray, scanner: Scanner, k: int) -> np.ndarray:
    """Reference depth for view k by trimesh ray casting through every pixel centre (slow)."""
    import trimesh

    r = rot_z(scanner.yaws[k])
    xc, zc = scanner.pixel_centres()
    origins = np.stack([xc.ravel(), np.full(xc.size, -scanner.distance - 1.0), zc.ravel()], 1) @ r
    dirs = np.tile([0.0, 1.0, 0.0], (xc.size, 1)) @ r
    loc, ray, _ = trimesh.Trimesh(v, f, process=False).ray.intersects_location(origins, dirs, multiple_hits=True)
    out = np.full(xc.size, BACKGROUND)
    np.minimum.at(out, ray, (loc @ r.T)[:, 1] + scanner.distance)
    return out.reshape(xc.shape)


def draw_views(s: dict, scanner: Scanner, title: str, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col, row, _ = scanner.project(s["landmarks"])
    fig, axes = plt.subplots(1, scanner.n_views, figsize=(2 * scanner.n_views, 3.6))
    for k, ax in enumerate(axes):
        d = np.where(np.isfinite(s["depth"][k]), s["depth"][k], np.nan)
        ax.imshow(d, cmap="viridis_r")
        vis = s["visible"][k]
        ax.scatter(col[k][vis], row[k][vis], s=6, c="red")
        ax.scatter(col[k][~vis], row[k][~vis], s=6, facecolors="none", edgecolors="white", linewidths=0.5)
        ax.set_title(f"{scanner.yaws[k]:.0f} deg", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> None:
    """Acceptance checks A1/A1b/A2 -> results/scanner_check.json, plus a figure of noisy views."""
    import json
    import time
    from pathlib import Path

    from body3d import bodies, landmarks

    sc, f, idx = Scanner(), bodies.faces(), landmarks.load()
    results = Path(__file__).parents[2] / "results"

    b = bodies.sample_bodies(5, seed=11)
    sc.render(b.vertices[0], f)
    t = time.perf_counter()
    renders = [sc.render(v, f)[0] for v in b.vertices]
    ms = (time.perf_counter() - t) / len(b) * 1e3
    diffs, only_one = [], 0
    for v, d in zip(b.vertices, renders):
        for k in range(sc.n_views):
            ref = ray_cast_depth(v, f, sc, k)
            both = np.isfinite(ref) & np.isfinite(d[k])
            only_one += int((np.isfinite(ref) != np.isfinite(d[k])).sum())
            diffs.append(np.abs(ref[both] - d[k][both]) * 1e3)
    diffs = np.concatenate(diffs)

    test = bodies.sample_stratified(400, seed=3)
    rng = np.random.default_rng(0)
    scans = [scan(v, f, v[idx], rng, sc, noise=None) for v in test.vertices]
    n_vis = np.array([s["visible"].sum(0) for s in scans])
    n_obs = np.array([s["observable"].sum(0) for s in scans])
    ok = (n_obs >= 2).all(1)
    short = n_obs < 2  # (bodies, landmarks)
    check = {
        "A1_ray_cast": {"bodies": 5, "views": sc.n_views, "pixels_compared": int(diffs.size),
                        "median_abs_mm": float(np.median(diffs)), "within_3mm_pct": float(np.mean(diffs <= 3) * 100),
                        "max_abs_mm": float(diffs.max()), "pixels_hit_by_only_one": only_one},
        "A1b_render_ms_per_body": round(ms, 1),
        "A2_observability": {
            "bodies": len(test),
            "definition": "observable = depth-visible within 1 cm, or within 1 px of the silhouette",
            "pct_bodies_all_landmarks_observable_in_2plus_views": float(ok.mean() * 100),
            "pct_bodies_short_per_landmark": {n: round(float(short[:, j].mean() * 100), 2) for j, n in enumerate(landmarks.NAMES)},
            "pct_bodies_short_per_bmi_band": {l: round(float((~ok[test.bmi_band == i]).mean() * 100), 2)
                                              for i, l in enumerate(bodies.BMI_LABELS)},
            "mean_observable_views": dict(zip(landmarks.NAMES, np.round(n_obs.mean(0), 2).tolist())),
            "mean_depth_visible_views": dict(zip(landmarks.NAMES, np.round(n_vis.mean(0), 2).tolist())),
        },
        "scanner": {"views": sc.n_views, "image": [sc.height, sc.width], "pixel_mm": sc.pixel * 1e3},
    }
    (results / "scanner_check.json").write_text(json.dumps(check, indent=1) + "\n")
    print(json.dumps(check, indent=1))

    heavy = int(np.argmax(test.bmi))
    s = scan(test.vertices[heavy], f, test.vertices[heavy][idx], np.random.default_rng(1), sc,
             sway=Sway(), arm_w=bodies.arm_weights())
    draw_views(s, sc, f"SYNTHETIC scan, mesh BMI {test.bmi[heavy]:.1f}, height {test.height[heavy]:.2f} m "
                      "(red = visible landmark, white ring = occluded)", results / "scanner_views.png")


if __name__ == "__main__":
    main()
