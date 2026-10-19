"""Fixes for a narrow training distribution (P1d). SYNTHETIC data throughout.

E2, ``Inflate`` (model-free): offset the mesh along its vertex normals by a smooth,
region-varying amount, so a body gains (or loses) soft tissue where people do. The
landmarks move with their vertices. It needs only a body-part segmentation, which a
real pipeline gets from its registered template, so it also applies to labelled real scans.

E3, ``extrapolate`` (model-based): re-generate the same training subjects with the
adiposity controls pushed past the training range. Optimistic here, because the test
bodies come from the same generator.
"""

from __future__ import annotations

import numpy as np

# region -> (bone-name substrings, offset range in mm along the outward normal)
REGIONS = {
    "abdomen": (("spine05", "spine04", "spine03"), (-10.0, 60.0)),
    "chest": (("spine02", "spine01"), (-10.0, 35.0)),
    "hips": (("root", "pelvis"), (-10.0, 50.0)),
    "thighs": (("upperleg",), (-10.0, 45.0)),
    "calves": (("lowerleg",), (-5.0, 20.0)),
    "upper_arms": (("upperarm", "shoulder"), (-5.0, 25.0)),
    "forearms": (("lowerarm",), (-3.0, 10.0)),
    "neck": (("neck",), (-5.0, 20.0)),
}
P_APPLY = 0.8  # fraction of training samples that get inflated
REGION_SPREAD = 0.25  # how far each region strays from the body-wide level (0-1 scale)


def region_weights() -> np.ndarray:
    """(V, R) skinning weight of every vertex in each region (head, hands, feet get none)."""
    from body3d.bodies import get_model

    m = get_model()
    w = m.vertex_bone_weights.cpu().numpy()
    bi = m.vertex_bone_indices.cpu().numpy()
    out = np.zeros((len(w), len(REGIONS)), np.float32)
    for r, (keys, _) in enumerate(REGIONS.values()):
        ids = [i for i, label in enumerate(m.bone_labels) if any(k in label for k in keys)]
        out[:, r] = (w * np.isin(bi, ids)).sum(1)
    return out


def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals (outward for Anny's winding)."""
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for j in range(3):
        np.add.at(vn, f[:, j], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


class Inflate:
    """Callable ``(vertices, rng) -> vertices`` used by the training dataset (picklable)."""

    __name__ = "inflate"

    def __init__(self, faces: np.ndarray, weights: np.ndarray, p_apply: float = P_APPLY):
        self.faces, self.weights, self.p_apply = faces, weights, p_apply
        self.lo = np.array([lo for _, (lo, _) in REGIONS.values()]) * 1e-3
        self.hi = np.array([hi for _, (_, hi) in REGIONS.values()]) * 1e-3

    def offsets(self, rng: np.random.Generator) -> np.ndarray:
        """Per-region offsets in metres: one body-wide level plus correlated regional spread."""
        level = rng.uniform(0.0, 1.0)
        frac = np.clip(level + rng.normal(0.0, REGION_SPREAD, len(self.lo)), 0.0, 1.0)
        return self.lo + (self.hi - self.lo) * frac

    def __call__(self, v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() >= self.p_apply:
            return v
        delta = self.weights @ self.offsets(rng)
        return v + delta[:, None] * vertex_normals(v, self.faces)


# E3: which local controls carry adiposity, and how far they are pushed
ADIPOSITY_LOCALS = ("measure-waist-circ-incr", "measure-hips-circ-incr", "measure-bust-circ-incr",
                    "measure-thigh-circ-incr", "stomach-pregnant-incr", "buttocks-volume-incr",
                    "l-upperleg-fat-incr")
P_HEAVIER = 0.8


def extrapolate(phenotype: np.ndarray, local: np.ndarray, rng: np.random.Generator):
    """Same subjects (height, sex, age, proportions, muscle, pose kept), adiposity pushed:
    80% heavier (weight +0.3..0.8, adiposity controls +0..0.6), 20% lighter.
    Anny's weight control can also change stature for some bodies; ``data`` corrects
    the height control afterwards so stature is kept."""
    from body3d.bodies import LOCAL_NAMES, get_model

    ph, lc = phenotype.copy(), local.copy()
    n = len(ph)
    heavier = rng.random(n) < P_HEAVIER
    wi = list(get_model().phenotype_labels).index("weight")
    dw = np.where(heavier, rng.uniform(0.3, 0.8, n), -rng.uniform(0.1, 0.4, n))
    ph[:, wi] = np.clip(ph[:, wi] + dw, 0.0, 1.0)
    cols = [LOCAL_NAMES.index(name) for name in ADIPOSITY_LOCALS]
    dl = np.where(heavier[:, None], rng.uniform(0.0, 0.6, (n, len(cols))), -rng.uniform(0.0, 0.3, (n, len(cols))))
    lc[:, cols] = np.clip(lc[:, cols] + dl, -1.0, 1.0)
    return ph, lc
