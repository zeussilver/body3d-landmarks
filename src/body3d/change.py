"""Body-shape change over time from meshes / STL (SYNTHETIC trajectories, illustrative only).

- ``make_trajectories``: 6 synthetic subjects x 4 scans (weeks 0, 8, 16, 24) losing about
  10% of mesh mass along a fast-then-plateau curve, re-positioned on the turntable at
  every visit, exported as binary STL in millimetres.
- Change maps, two paths:
  A. same topology (vertex correspondence known): Kabsch on stable regions (head, shins),
     then signed displacement along the week-0 vertex normals;
  B. arbitrary STL (no correspondence): point-to-plane ICP weighted towards the same
     height bands, then signed distance to the nearest vertex's tangent plane.
- Girths at waist / hip / thigh by mesh-plane slicing, at heights from landmarks
  predicted by the phase-1 model on each simulated scan.

    python -m body3d.change      # writes viewer/data/* and results/change_check.json
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from body3d import bodies, landmarks, stl
from body3d.augment import ADIPOSITY_LOCALS, vertex_normals
from body3d.scanner import rot_z

WEEKS = (0, 8, 16, 24)
TARGET_MASS_LOSS = 0.10  # fraction of week-0 mesh mass lost by week 24
TAU_WEEKS = 10.0  # loss curve time constant
N_SUBJECTS = 6
SUBJECT_SEED = 404
START_BMI = (30.0, 42.0)
STABLE_TOP, STABLE_BOTTOM = 0.12, 0.30  # head and shin bands, as fractions of stature
DISPLAY_RANGE_MM = 20.0  # colour scale; week-24 changes span about -24..+10 mm
ARM_MASK = 0.5  # arm skinning weight above which the change map is greyed out (pose dominates)
THIGH_BELOW_CROTCH = 0.05  # m

N = {n: i for i, n in enumerate(landmarks.NAMES)}


# ---------------------------------------------------------------- trajectories

def loss_curve(week: float) -> float:
    """Fraction of the final loss reached at ``week`` (0 at week 0, 1 at week 24)."""
    return (1 - np.exp(-week / TAU_WEEKS)) / (1 - np.exp(-WEEKS[-1] / TAU_WEEKS))


def pick_subjects(n: int = N_SUBJECTS, seed: int = SUBJECT_SEED) -> bodies.Bodies:
    """Half male, half female, starting mesh BMI 30-42, spread over stature."""
    pool = bodies.sample_bodies(600, seed, bmi_range=START_BMI)
    chosen = []
    for sex in (0, 1):
        idx = np.flatnonzero(pool.sex == sex)
        idx = idx[np.argsort(pool.height[idx])]
        chosen += idx[np.linspace(0, len(idx) - 1, n // 2).round().astype(int)].tolist()
    return pool.subset(np.array(chosen))


def _params(subj: bodies.Bodies, amount: np.ndarray, frac: float):
    """Phenotype/local controls after ``frac`` of a loss of size ``amount`` (per subject)."""
    ph, lc = np.array(subj.phenotype), np.array(subj.local)
    wi = list(bodies.get_model().phenotype_labels).index("weight")
    ph[:, wi] = np.clip(ph[:, wi] - frac * amount, 0, 1)
    cols = [bodies.LOCAL_NAMES.index(c) for c in ADIPOSITY_LOCALS]
    lc[:, cols] = np.clip(lc[:, cols] - frac * 0.6 * amount[:, None], -1, 1)
    return ph, lc


def _mass(b: bodies.Bodies) -> np.ndarray:
    return b.bmi * b.height**2  # Anny: volume x 980 kg/m^3


def calibrate(subj: bodies.Bodies, iters: int = 12) -> np.ndarray:
    """Per-subject loss size giving TARGET_MASS_LOSS at week 24 (bisection, stature kept)."""
    lo, hi = np.zeros(len(subj)), np.full(len(subj), 1.5)
    m0 = _mass(subj)
    for _ in range(iters):
        mid = (lo + hi) / 2
        ph, lc = _params(subj, mid, 1.0)
        ph = bodies.keep_stature(ph, lc, subj.pose, subj.height)
        loss = 1 - _mass(bodies.build(ph, lc, subj.pose)) / m0
        lo, hi = np.where(loss < TARGET_MASS_LOSS, mid, lo), np.where(loss < TARGET_MASS_LOSS, hi, mid)
    return (lo + hi) / 2


def make_trajectories(seed: int = SUBJECT_SEED):
    """Body-frame meshes per subject and week, plus the per-visit turntable placements."""
    subj = pick_subjects(seed=seed)
    amount = calibrate(subj)
    rng = np.random.default_rng(seed)
    weeks = []
    for w in WEEKS:
        pose = np.array(subj.pose)
        pose[:, :2] += rng.uniform(-3, 3, (len(subj), 2))  # arms held a little differently each visit
        pose[:, 2] += rng.uniform(-1, 1, len(subj))
        ph, lc = _params(subj, amount, loss_curve(w))
        ph = bodies.keep_stature(ph, lc, pose, subj.height)
        weeks.append(bodies.build(ph, lc, pose))
    placements = [[(rot_z(rng.uniform(-15, 15)), np.r_[rng.uniform(-0.02, 0.02, 2), 0.0]) for _ in WEEKS]
                  for _ in range(len(subj))]
    return subj, weeks, placements


# ---------------------------------------------------------------- alignment

def kabsch(p: np.ndarray, q: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rigid R, t minimising sum w |R p + t - q|^2."""
    w = np.ones(len(p)) if w is None else w
    pm, qm = (w[:, None] * p).sum(0) / w.sum(), (w[:, None] * q).sum(0) / w.sum()
    h = ((p - pm) * w[:, None]).T @ (q - qm)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, qm - r @ pm


def _rodrigues(omega: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(omega)
    if th < 1e-12:
        return np.eye(3)
    k = omega / th
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * kx @ kx


def stable_weights(v: np.ndarray, low: float = 0.05) -> np.ndarray:
    """1 for head and shin height bands (little change with weight), ``low`` elsewhere."""
    z = v[:, 2]
    h = z.max() - z.min()
    return np.where((z > z.max() - STABLE_TOP * h) | (z < z.min() + STABLE_BOTTOM * h), 1.0, low)


def icp(src: np.ndarray, dst: np.ndarray, dst_n: np.ndarray, w: np.ndarray | None = None, iters: int = 40,
        trim: float = 0.9, yaw_starts=(-40, -20, 0, 20, 40)) -> tuple[np.ndarray, np.ndarray, float]:
    """Point-to-plane ICP of ``src`` onto (``dst``, normals ``dst_n``); multi-start over yaw.

    Returns R, t (src -> dst) and the final weighted RMS point-to-plane residual in metres.
    """
    w = np.ones(len(src)) if w is None else w
    tree = cKDTree(dst)

    def run(r, t, n_it):
        for _ in range(n_it):
            p = src @ r.T + t
            _, j = tree.query(p)
            res = ((p - dst[j]) * dst_n[j]).sum(1)
            keep = np.abs(res) <= np.quantile(np.abs(res), trim)
            a = np.c_[np.cross(p[keep], dst_n[j][keep]), dst_n[j][keep]] * np.sqrt(w[keep])[:, None]
            x = np.linalg.lstsq(a, -res[keep] * np.sqrt(w[keep]), rcond=None)[0]
            dr = _rodrigues(x[:3])
            r, t = dr @ r, dr @ t + x[3:]
        p = src @ r.T + t
        _, j = tree.query(p)
        res = ((p - dst[j]) * dst_n[j]).sum(1)
        keep = np.abs(res) <= np.quantile(np.abs(res), trim)
        return r, t, float(np.sqrt((w[keep] * res[keep] ** 2).sum() / w[keep].sum()))

    best = None
    for yaw in yaw_starts:
        r0 = rot_z(yaw)
        t0 = dst.mean(0) - src.mean(0) @ r0.T
        cand = run(r0, t0, 8)
        if best is None or cand[2] < best[2]:
            best = cand
    return run(best[0], best[1], iters)


# ---------------------------------------------------------------- change maps

def topology_change(v0: np.ndarray, vt: np.ndarray, f: np.ndarray, arm_w: np.ndarray):
    """Path A. Returns vt aligned to v0 and signed displacement (V,) in metres along week-0
    normals (NaN on the arms, whose change is dominated by pose)."""
    w = stable_weights(v0, low=0.0) * (arm_w.sum(1) < 0.05)  # align on head and shins, away from the arms
    r, t = kabsch(vt, v0, w)
    aligned = vt @ r.T + t
    disp = ((aligned - v0) * vertex_normals(v0, f)).sum(1)
    disp[arm_w.sum(1) >= ARM_MASK] = np.nan
    return aligned, disp


def surface_change(v0: np.ndarray, f0: np.ndarray, vt: np.ndarray, ft: np.ndarray):
    """Path B, no correspondence. Returns vt aligned to v0, signed displacement at each v0
    vertex (metres) and the ICP residual."""
    n0 = vertex_normals(v0, f0)
    r, t, rms = icp(vt, v0, n0, w=stable_weights(vt))
    aligned = vt @ r.T + t
    nt = vertex_normals(aligned, ft)
    _, j = cKDTree(aligned).query(v0)
    return aligned, ((aligned[j] - v0) * nt[j]).sum(1), rms


# ---------------------------------------------------------------- girths

def slice_loops(v: np.ndarray, f: np.ndarray, z: float) -> list[np.ndarray]:
    """Closed cross-section polygons (each (M, 2) xy) of a watertight mesh at height ``z``.

    Intersection points are keyed by the mesh edge they lie on, so segments from
    neighbouring triangles chain exactly.
    """
    s = v[:, 2] - z
    s[s == 0] = 1e-12
    pos = s[f] > 0
    tris = f[pos.any(1) & ~pos.all(1)]
    links: dict[tuple[int, int], list[tuple[int, int]]] = {}
    points: dict[tuple[int, int], np.ndarray] = {}
    for tri in tris:
        keys = []
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if (s[a] > 0) != (s[b] > 0):
                key = (min(a, b), max(a, b))
                if key not in points:
                    points[key] = v[a, :2] + (v[b, :2] - v[a, :2]) * (s[a] / (s[a] - s[b]))
                keys.append(key)
        k1, k2 = keys
        links.setdefault(k1, []).append(k2)
        links.setdefault(k2, []).append(k1)
    loops, seen = [], set()
    for start in links:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [k for k in links[cur] if k != prev]
            if not nxt or nxt[0] == start:
                break
            prev, cur = cur, nxt[0]
            if cur in seen:
                break
            seen.add(cur)
            loop.append(cur)
        loops.append(np.array([points[k] for k in loop]))
    return loops


def polygon_area(p: np.ndarray) -> float:
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def perimeters(p: np.ndarray) -> tuple[float, float]:
    """(contour length, convex-hull length = what a tape measures), metres."""
    contour = float(np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1).sum())
    return contour, float(ConvexHull(p).area)


def girths(v: np.ndarray, f: np.ndarray, lm: np.ndarray) -> dict[str, dict[str, float]]:
    """Waist (waist-front height), hip (buttock-point height) and left thigh (5 cm below the
    crotch) girths in mm. Waist/hip take the largest loop (the trunk, not the arms); the
    thigh takes the loop furthest to the body's left."""
    left = lm[N["acromion_l"], :2] - lm[N["acromion_r"], :2]
    left /= np.linalg.norm(left)
    levels = {"waist": lm[N["waist_front"], 2], "hip": lm[N["buttock_point"], 2],
              "thigh": lm[N["crotch"], 2] - THIGH_BELOW_CROTCH}
    out = {}
    for name, z in levels.items():
        loops = [p for p in slice_loops(v, f, z) if len(p) >= 3]
        if name == "thigh":
            loop = max(loops, key=lambda p: (p.mean(0) - lm[N["crotch"], :2]) @ left)
        else:
            loop = max(loops, key=polygon_area)
        contour, tape = perimeters(loop)
        out[name] = {"z_m": float(z), "contour_mm": contour * 1e3, "tape_mm": tape * 1e3}
    return out


# ---------------------------------------------------------------- landmarks from the scanner

def predict_landmarks(meshes: list[np.ndarray], f: np.ndarray, checkpoint: Path, seed: int = 0) -> np.ndarray:
    """Phase-1 model on simulated scans of already-placed meshes -> (n, 16, 3), same frame."""
    import torch

    from body3d import data
    from body3d.model import LandmarkModel, depth_to_input
    from body3d.scanner import scan

    model = LandmarkModel()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    noise = dataclasses.replace(data.NOISE, facing_deg=0.0, offset_m=0.0)  # meshes are already placed
    idx, arm_w = landmarks.load(), bodies.arm_weights()
    out = []
    with torch.no_grad():
        for i, v in enumerate(meshes):
            s = scan(v, f, v[idx], np.random.default_rng([seed, i]), data.SCANNER, noise, data.SWAY, arm_w)
            out.append(model(depth_to_input(s["depth"])[None])["points"][0].numpy())
    return np.stack(out)


# ---------------------------------------------------------------- pipeline

def main() -> None:
    from body3d.data import RESULTS, ROOT

    viewer = ROOT / "viewer" / "data"
    raw_dir = ROOT / "data" / "trajectories"
    f, arm_w, idx = bodies.faces(), bodies.arm_weights(), landmarks.load()
    subj, weeks, placements = make_trajectories()

    # "scanner output": each visit's mesh in its own turntable frame, written as STL (mm)
    placed = [[weeks[k].vertices[i] @ r.T + t for k, (r, t) in enumerate(placements[i])] for i in range(len(subj))]
    for i in range(len(subj)):
        (raw_dir / f"s{i + 1}").mkdir(parents=True, exist_ok=True)
        for k, w in enumerate(WEEKS):
            stl.write(raw_dir / f"s{i + 1}" / f"week{w:02d}.stl", placed[i][k] * 1e3, f, "SYNTHETIC body3d units=mm")

    ckpt = ROOT / "runs" / "e0_s0" / "best.pt"
    pred = predict_landmarks([m for row in placed for m in row], f, ckpt).reshape(len(subj), len(WEEKS), 16, 3)

    manifest = {"synthetic": True, "units": "mm", "weeks": list(WEEKS), "display_range_mm": DISPLAY_RANGE_MM,
                "note": "SYNTHETIC Anny bodies on an illustrative weight-loss curve; not clinical data",
                "subjects": []}
    agree, icp_rms, level_effect = [], [], []
    for i in range(len(subj)):
        sid = f"s{i + 1}"
        (viewer / sid).mkdir(parents=True, exist_ok=True)
        v0_raw = placed[i][0]
        g_pred, g_gt, mass = [], [], []
        for k, w in enumerate(WEEKS):
            vt_raw = placed[i][k]
            g_pred.append(girths(vt_raw, f, pred[i, k]))
            g_gt.append(girths(vt_raw, f, vt_raw[idx]))
            mass.append(float(_mass(weeks[k].subset([i]))[0]))
            aligned, disp = topology_change(v0_raw, vt_raw, f, arm_w)
            if k > 0:
                # path B from the STL files alone (vertex order scrambled by welding)
                v0s, f0s = stl.read(raw_dir / sid / "week00.stl")
                vts, fts = stl.read(raw_dir / sid / f"week{w:02d}.stl")
                _, disp_b_s, rms = surface_change(v0s / 1e3, f0s, vts / 1e3, fts)
                j = cKDTree(v0s / 1e3).query(v0_raw)[1]  # map welded vertices back to Anny order
                ok = ~np.isnan(disp)
                agree.append(float(np.median(np.abs(disp[ok] - disp_b_s[j][ok])) * 1e3))
                icp_rms.append(rms * 1e3)
            stl.write(viewer / sid / f"week{w:02d}.stl", aligned * 1e3, f, "SYNTHETIC body3d units=mm aligned")
            (disp[f.ravel()] * 1e3).astype("<f4").tofile(viewer / sid / f"week{w:02d}.disp.f32")
        for gp, gg in zip(g_pred, g_gt):
            level_effect += [abs(gp[n]["tape_mm"] - gg[n]["tape_mm"]) for n in gp]
        manifest["subjects"].append({
            "id": sid, "sex": bodies.SEX_LABELS[int(subj.sex[i])], "height_m": round(float(subj.height[i]), 3),
            "mesh_bmi": [round(float(weeks[k].bmi[i]), 1) for k in range(len(WEEKS))],
            "mass_kg": [round(m, 1) for m in mass],
            "mass_change_pct": [round((m / mass[0] - 1) * 100, 1) for m in mass],
            "girths_mm": {n: {"tape": [round(g[n]["tape_mm"], 1) for g in g_pred],
                              "contour": [round(g[n]["contour_mm"], 1) for g in g_pred]} for n in g_pred[0]},
        })
    viewer.mkdir(parents=True, exist_ok=True)
    (viewer / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    check = {
        "data": "SYNTHETIC",
        "subjects": len(subj), "weeks": list(WEEKS),
        "mass_change_week24_pct": [s["mass_change_pct"][-1] for s in manifest["subjects"]],
        "A5_path_agreement_median_abs_mm": {"per_scan": [round(a, 2) for a in agree], "max": max(agree)},
        "icp_rms_mm": {"mean": float(np.mean(icp_rms)), "max": float(np.max(icp_rms))},
        "girth_level_effect_tape_mm": {"mean": float(np.mean(level_effect)), "max": float(np.max(level_effect)),
                                       "note": "predicted vs ground-truth landmark heights"},
        "girth_change_week24_tape_mm": {n: [round(s["girths_mm"][n]["tape"][-1] - s["girths_mm"][n]["tape"][0], 1)
                                            for s in manifest["subjects"]] for n in ("waist", "hip", "thigh")},
    }
    (RESULTS / "change_check.json").write_text(json.dumps(check, indent=1) + "\n")
    print(json.dumps(check, indent=1))


if __name__ == "__main__":
    main()
