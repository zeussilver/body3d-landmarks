"""16 anthropometric landmarks (in the spirit of ISO 8559-1) as fixed Anny vertex indices.

Indices are chosen once on the neutral template (``bodies.template()``) with the
geometric rules below, then frozen in ``landmarks.json``. Anny keeps its topology
fixed, so the same index is the same anatomical point on every generated body.

Run ``python -m body3d.landmarks`` to re-select, rewrite the JSON and redraw
``results/landmarks_template.png``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

JSON_PATH = Path(__file__).with_name("landmarks.json")

NAMES = (
    "vertex", "cervicale", "acromion_l", "acromion_r", "bust_point_l", "bust_point_r",
    "waist_front", "waist_back", "waist_side_l", "waist_side_r", "buttock_point",
    "crotch", "knee_l", "knee_r", "ankle_l", "ankle_r",
)

RULES = {
    "vertex": "highest vertex of the head",
    "cervicale": "most posterior midline vertex within 2 cm of the neck01 joint height (C7 proxy)",
    "acromion_l": "highest vertex within 1.5 cm (x) / 3 cm (y) of the left shoulder joint",
    "bust_point_l": "most anterior non-arm vertex, x 3-20 cm, z 68-78% of stature",
    "waist_front": "front midline vertex at natural-waist height (min non-arm convex-hull perimeter, z 57-70%)",
    "waist_back": "back midline vertex at natural-waist height",
    "waist_side_l": "most lateral non-arm vertex within 1 cm of natural-waist height and 3 cm of mid-depth",
    "buttock_point": "most posterior non-arm vertex of the left buttock, z 40-56% (max protrusion)",
    "crotch": "lowest midline vertex of the trunk",
    "knee_l": "most anterior vertex within 2 cm of the knee joint height (mid-patella proxy)",
    "ankle_l": "most lateral leg vertex within 1.5 cm of the ankle joint (lateral malleolus proxy)",
}
MIRRORS = {"acromion_r": "acromion_l", "bust_point_r": "bust_point_l", "waist_side_r": "waist_side_l",
           "knee_r": "knee_l", "ankle_r": "ankle_l"}

_ARM_BONES = ("shoulder", "upperarm", "lowerarm", "wrist", "finger", "metacarpal")
_LEG_BONES = ("upperleg", "lowerleg", "foot", "toe")
MIDLINE_TOL = 2e-3  # m


def load() -> np.ndarray:
    """Frozen landmark vertex indices in NAMES order."""
    data = json.loads(JSON_PATH.read_text())
    by_name = {d["name"]: d["index"] for d in data["landmarks"]}
    return np.array([by_name[n] for n in NAMES], dtype=np.int64)


def positions(vertices: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
    """(B, V, 3) or (V, 3) vertices -> (..., 16, 3) landmark positions."""
    return vertices[..., load() if idx is None else idx, :]


def dominant_bone_masks() -> tuple[np.ndarray, np.ndarray]:
    """Boolean (V,) masks of vertices driven mainly by arm bones and by leg bones."""
    from body3d.bodies import get_model

    m = get_model()
    w = m.vertex_bone_weights.cpu().numpy()
    bi = m.vertex_bone_indices.cpu().numpy()
    dom = bi[np.arange(len(bi)), w.argmax(1)]
    labels = np.array(m.bone_labels)[dom]
    arm = np.array([any(k in l for k in _ARM_BONES) for l in labels])
    leg = np.array([any(k in l for k in _LEG_BONES) for l in labels])
    return arm, leg


def _waist_height(v: np.ndarray, body: np.ndarray, h: float) -> float:
    best = (np.inf, 0.0)
    for z in np.arange(0.57 * h, 0.70 * h, 0.005):
        pts = v[body & (np.abs(v[:, 2] - z) < 0.01)][:, :2]
        if len(pts) >= 8:
            best = min(best, (ConvexHull(pts).area, z))  # 2D hull "area" is its perimeter
    return best[1]


def select(v: np.ndarray, heads: dict[str, np.ndarray]) -> dict[str, int]:
    """Apply RULES to template vertices (V, 3); returns name -> vertex index."""
    arm, leg = dominant_bone_masks()
    trunk = ~arm & ~leg
    h = v[:, 2].max()
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    midline = np.abs(x) < MIDLINE_TOL
    idx = np.arange(len(v))

    def pick(mask, score):  # index of the masked vertex with the largest score
        return int(idx[mask][np.argmax(score[mask])])

    out = {"vertex": pick(np.ones(len(v), bool), z - 10 * np.abs(x))}
    out["cervicale"] = pick(midline & (np.abs(z - heads["neck01"][2]) < 0.02), y)
    j = heads["upperarm01.L"]
    out["acromion_l"] = pick((np.abs(x - j[0]) < 0.015) & (np.abs(y - j[1]) < 0.03), z)
    out["bust_point_l"] = pick(~arm & (x > 0.03) & (x < 0.20) & (z > 0.68 * h) & (z < 0.78 * h), -y)
    zw = _waist_height(v, ~arm, h)
    out["waist_front"] = pick(midline & (y < heads["root"][1]), -np.abs(z - zw))
    out["waist_back"] = pick(midline & (y > heads["root"][1]), -np.abs(z - zw))
    y_mid = (y[out["waist_front"]] + y[out["waist_back"]]) / 2
    out["waist_side_l"] = pick(~arm & (x > 0) & (np.abs(z - zw) < 0.01) & (np.abs(y - y_mid) < 0.03), x)
    out["buttock_point"] = pick(~arm & (x > 0) & (z > 0.40 * h) & (z < 0.56 * h), y)
    out["crotch"] = pick(midline & trunk & (z > 0.3 * h), -z)
    j = heads["lowerleg01.L"]
    out["knee_l"] = pick(leg & (np.abs(z - j[2]) < 0.02) & (np.abs(x - j[0]) < 0.04), -y)
    j = heads["foot.L"]
    out["ankle_l"] = pick(leg & (np.abs(z - j[2]) < 0.015) & (np.abs(y - j[1]) < 0.03), x)

    tree = cKDTree(v)
    for right, left in MIRRORS.items():
        out[right] = int(tree.query(v[out[left]] * np.array([-1.0, 1.0, 1.0]))[1])
    return {n: out[n] for n in NAMES}


def mirror_residuals_mm(v: np.ndarray, sel: dict[str, int]) -> dict[str, float]:
    return {r: float(np.linalg.norm(v[sel[r]] * [-1, 1, 1] - v[sel[l]]) * 1e3) for r, l in MIRRORS.items()}


def draw(v: np.ndarray, sel: dict[str, int], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 8))
    for ax, (a, title) in zip(axes, ((0, "front (x-z)"), (1, "side (y-z)"))):
        ax.scatter(v[:, a] if a == 0 else -v[:, 1], v[:, 2], s=0.2, c="0.75")
        for n, i in sel.items():
            px = v[i, 0] if a == 0 else -v[i, 1]
            ax.scatter(px, v[i, 2], s=18, c="crimson")
            ax.annotate(n, (px, v[i, 2]), fontsize=6, xytext=(3, 2), textcoords="offset points")
        ax.set_aspect("equal")
        ax.set_title(title)
    fig.suptitle("Landmarks on the neutral Anny template (SYNTHETIC)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main() -> None:
    from body3d import bodies

    t = bodies.template()
    v = t.vertices[0]
    sel = select(v, bodies.bone_heads(t))
    res = mirror_residuals_mm(v, sel)
    JSON_PATH.write_text(json.dumps({
        "model": "anny==0.6.1, default 'anny' topology (13,718 vertices)",
        "template": "bodies.template(): all phenotypes 0.5, morphological age 35, no local controls, no pose",
        "frame": "metres, z up, faces -y, left = +x",
        "landmarks": [{"name": n, "index": sel[n], "rule": RULES.get(n, f"mirror of {MIRRORS.get(n)}")} for n in NAMES],
    }, indent=1) + "\n")
    draw(v, sel, Path(__file__).parents[2] / "results" / "landmarks_template.png")
    for n in NAMES:
        print(f"{n:14s} {sel[n]:6d}  {np.round(v[sel[n]], 3)}")
    print("mirror residuals (mm):", {k: round(r, 2) for k, r in res.items()})


if __name__ == "__main__":
    main()
