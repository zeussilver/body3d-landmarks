"""Synthetic adult bodies from the Anny parametric body model.

SYNTHETIC DATA: every body here is generated, none comes from a real person.

Coordinates: metres, z up, feet at z = 0, the root (pelvis) joint at x = y = 0,
the body faces -y and its left side is +x. Only Anny's default ``anny`` topology is used (CC0 assets);
the non-commercial ``smplx``/``smpl`` topologies are never requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

BMI_EDGES = (18.5, 25.0, 30.0, 35.0)
BMI_LABELS = ("<18.5", "18.5-25", "25-30", "30-35", ">=35")
HEIGHT_EDGES = (1.60, 1.75)
HEIGHT_LABELS = ("<1.60", "1.60-1.75", ">=1.75")
SEX_LABELS = ("male", "female")
# Bodies outside this mesh-BMI range are rejected everywhere: the local controls can
# otherwise produce implausibly thin adults (mesh BMI down to ~11).
BMI_PLAUSIBLE = (15.0, 50.0)

# Local shape controls added on top of Anny's WHO-calibrated phenotype sampler, as
# (control names, std of a N(0, std) draw clipped to [-1, 1]). L/R controls share one draw.
LOCAL_CONTROLS = (
    (("measure-waist-circ-incr",), 0.3),
    (("measure-hips-circ-incr",), 0.3),
    (("measure-bust-circ-incr",), 0.3),
    (("measure-thigh-circ-incr",), 0.3),
    (("stomach-pregnant-incr",), 0.25),
    (("buttocks-volume-incr",), 0.3),
    (("l-upperleg-fat-incr", "r-upperleg-fat-incr"), 0.3),
    (("torso-scale-depth-incr",), 0.2),
    (("measure-shoulder-dist-incr",), 0.2),
)
LOCAL_NAMES = tuple(group[0] for group, _ in LOCAL_CONTROLS)

# Pose jitter in degrees: arm adduction per side (+ = arm closer to the body) and stance
# (+ = feet further apart). Applied as rotations about the front-back (y) axis.
ARM_JITTER_DEG = 10.0
STANCE_JITTER_DEG = (-3.0, 5.0)
POSE_NAMES = ("arm_l", "arm_r", "stance")

ADULT_AGE_RANGE = (20.0, 70.0)  # morphological age, years
TEMPLATE_AGE = 35.0


@lru_cache(maxsize=1)
def get_model():
    """Anny model (default triangulated topology, local controls enabled), CPU float32."""
    import anny

    return anny.Anny(local_changes="default").to(dtype=torch.float32)


@lru_cache(maxsize=1)
def _shape_distribution():
    from anny.shape_distribution import SimpleShapeDistribution

    model = get_model()
    age = torch.distributions.Uniform(torch.tensor(ADULT_AGE_RANGE[0]), torch.tensor(ADULT_AGE_RANGE[1]))
    return SimpleShapeDistribution(model, morphological_age_distribution=age, gender_distribution=_BinarySex())


class _BinarySex:
    """Gender drawn from [0, 0.2] (male end) or [0.8, 1] (female end) with equal odds.

    Anny's gender is a continuous blend; we keep the ends so 'sex' is a clean
    reporting stratum. Anny's own sampler treats gender <= 0.5 as male.
    """

    def sample(self, shape):
        female = torch.rand(shape) < 0.5
        u = torch.rand(shape) * 0.2
        return torch.where(female, 1.0 - u, u)


def faces() -> np.ndarray:
    return get_model().faces.cpu().numpy().astype(np.int64)


_ARM_CHAIN = ("upperarm", "lowerarm", "wrist", "finger", "metacarpal")


@lru_cache(maxsize=1)
def arm_weights() -> np.ndarray:
    """(V, 2) total skinning weight of the left / right arm chain per vertex (same for every body)."""
    m = get_model()
    w = m.vertex_bone_weights.cpu().numpy()
    bi = m.vertex_bone_indices.cpu().numpy()
    out = np.zeros((len(w), 2), np.float32)
    for side, suffix in enumerate((".L", ".R")):
        ids = [i for i, l in enumerate(m.bone_labels) if l.endswith(suffix) and any(k in l for k in _ARM_CHAIN)]
        out[:, side] = (w * np.isin(bi, ids)).sum(1)
    return out


@dataclass
class Bodies:
    vertices: np.ndarray  # (B, V, 3) float32
    phenotype: np.ndarray  # (B, 6) in get_model().phenotype_labels order
    local: np.ndarray  # (B, len(LOCAL_NAMES))
    pose: np.ndarray  # (B, 3) degrees, POSE_NAMES order
    height: np.ndarray  # (B,) stature in metres, rest pose
    bmi: np.ndarray  # (B,) mesh BMI: Anny volume x 980 kg/m^3 / height^2 (a proxy)

    def __len__(self) -> int:
        return len(self.vertices)

    @property
    def sex(self) -> np.ndarray:
        gender = self.phenotype[:, list(get_model().phenotype_labels).index("gender")]
        return (gender > 0.5).astype(np.int64)

    @property
    def bmi_band(self) -> np.ndarray:
        return np.digitize(self.bmi, BMI_EDGES)

    @property
    def height_band(self) -> np.ndarray:
        return np.digitize(self.height, HEIGHT_EDGES)

    def subset(self, idx) -> "Bodies":
        return Bodies(*(getattr(self, f)[idx] for f in self.__dataclass_fields__))

    @staticmethod
    def concat(parts: list["Bodies"]) -> "Bodies":
        return Bodies(*(np.concatenate([getattr(p, f) for p in parts]) for f in Bodies.__dataclass_fields__))

    def save(self, path) -> None:
        np.savez(path, **{f: getattr(self, f) for f in self.__dataclass_fields__})

    @staticmethod
    def load(path) -> "Bodies":
        with np.load(path) as d:
            return Bodies(*(d[f] for f in Bodies.__dataclass_fields__))


def _rot_y(deg: torch.Tensor) -> torch.Tensor:
    """(B,) degrees -> (B, 4, 4) homogeneous rotation about the y axis."""
    a = torch.deg2rad(deg.float())
    c, s = torch.cos(a), torch.sin(a)
    t = torch.eye(4).repeat(len(deg), 1, 1)
    t[:, 0, 0], t[:, 0, 2], t[:, 2, 0], t[:, 2, 2] = c, s, -s, c
    return t


def _pose_dict(pose: np.ndarray) -> dict[str, torch.Tensor]:
    # +y rotation swings the left (+x) arm inward and the left leg inward; mirror for the right.
    p = torch.from_numpy(pose)
    return {
        "upperarm01.L": _rot_y(p[:, 0]),
        "upperarm01.R": _rot_y(-p[:, 1]),
        "upperleg01.L": _rot_y(-p[:, 2]),
        "upperleg01.R": _rot_y(p[:, 2]),
    }


def _local_tensor(local: np.ndarray) -> torch.Tensor:
    """(B, len(LOCAL_NAMES)) control values -> (B, n_local_changes) tensor for Anny."""
    labels = list(get_model().local_change_labels)
    lc = torch.zeros(len(local), len(labels))
    for j, (group, _) in enumerate(LOCAL_CONTROLS):
        for name in group:
            lc[:, labels.index(name)] = torch.from_numpy(np.ascontiguousarray(local[:, j])).float()
    return lc


def build(phenotype: np.ndarray, local: np.ndarray, pose: np.ndarray, chunk: int = 256) -> Bodies:
    """Run Anny for given parameters; returns centred bodies with stature and mesh BMI."""
    from anny.anthropometry import Anthropometry

    model = get_model()
    anthro = Anthropometry(model)
    verts, heights, bmis = [], [], []
    with torch.no_grad():
        for s in range(0, len(phenotype), chunk):
            out = model(
                pose_parameters=_pose_dict(pose[s : s + chunk]),
                phenotype_kwargs=torch.from_numpy(phenotype[s : s + chunk]).float(),
                local_changes_kwargs=_local_tensor(local[s : s + chunk]),
            )
            rest = out["rest_vertices"]
            heights.append(anthro.height(rest).numpy())
            bmis.append(anthro.bmi(rest).numpy())
            v = out["vertices"].numpy()
            v[..., 2] -= v[..., 2].min(axis=1, keepdims=True)
            v[..., :2] -= out["bone_poses"][:, None, 0, :2, 3].numpy()
            verts.append(v.astype(np.float32))
    return Bodies(
        np.concatenate(verts), phenotype.astype(np.float32), local.astype(np.float32),
        pose.astype(np.float32), np.concatenate(heights), np.concatenate(bmis),
    )


def keep_stature(phenotype: np.ndarray, local: np.ndarray, pose: np.ndarray, target_height: np.ndarray,
                 iters: int = 3) -> np.ndarray:
    """Adjust the height control (Newton steps) so each body keeps ``target_height``.

    Anny's weight control can shift stature for some bodies; changing weight alone would
    otherwise make a different person.
    """
    ph = np.array(phenotype, dtype=np.float32)
    hi = list(get_model().phenotype_labels).index("height")
    for _ in range(iters):
        h = build(ph, local, pose).height
        probe = ph.copy()
        probe[:, hi] = np.clip(probe[:, hi] + 0.02, 0, 1)
        slope = (build(probe, local, pose).height - h) / 0.02
        ph[:, hi] = np.clip(ph[:, hi] - (h - target_height) / np.where(np.abs(slope) > 1e-3, slope, 1.0), 0, 1)
    return ph


def sample_params(n: int, seed: int, pose_jitter: bool = True, local_controls: bool = True):
    """Phenotype from Anny's WHO-calibrated adult sampler, plus local controls and pose jitter."""
    model = get_model()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        _, ph = _shape_distribution().sample(n)
    phenotype = torch.stack([ph[k] for k in model.phenotype_labels], dim=1).float().numpy()
    rng = np.random.default_rng(seed)
    stds = np.array([std for _, std in LOCAL_CONTROLS])
    local = np.clip(rng.normal(0.0, stds, size=(n, len(stds))), -1.0, 1.0) if local_controls else np.zeros((n, len(stds)))
    pose = np.zeros((n, 3))
    if pose_jitter:
        pose[:, :2] = rng.uniform(-ARM_JITTER_DEG, ARM_JITTER_DEG, size=(n, 2))
        pose[:, 2] = rng.uniform(*STANCE_JITTER_DEG, size=n)
    return phenotype, local.astype(np.float32), pose.astype(np.float32)


def sample_bodies(n: int, seed: int, bmi_range: tuple[float, float] | None = None,
                  chunk: int = 512, **kw) -> Bodies:
    """n plausible bodies, optionally restricted to a mesh-BMI range, by rejection sampling."""
    lo, hi = BMI_PLAUSIBLE
    if bmi_range is not None:
        lo, hi = max(lo, bmi_range[0]), min(hi, bmi_range[1])
    parts, have, i = [], 0, 0
    while have < n:
        b = build(*sample_params(chunk, seed * 100_003 + i, **kw))
        b = b.subset((b.bmi >= lo) & (b.bmi < hi))
        parts.append(b)
        have += len(b)
        i += 1
    return Bodies.concat(parts).subset(slice(0, n))


def sample_stratified(n_per_band: int, seed: int, chunk: int = 1024, max_chunks: int = 200, **kw) -> Bodies:
    """Equal counts per mesh-BMI band, each band split evenly by sex (plausible bodies only)."""
    per_cell = n_per_band // 2
    cells: dict[tuple[int, int], list[int]] = {(b, s): [] for b in range(len(BMI_LABELS)) for s in range(2)}
    parts, offset = [], 0
    for i in range(max_chunks):
        b = build(*sample_params(chunk, seed * 100_003 + i, **kw))
        b = b.subset((b.bmi >= BMI_PLAUSIBLE[0]) & (b.bmi < BMI_PLAUSIBLE[1]))
        keep = []
        for j, cell in enumerate(zip(b.bmi_band, b.sex)):
            if len(cells[cell]) < per_cell:
                cells[cell].append(offset + len(keep))
                keep.append(j)
        parts.append(b.subset(np.array(keep, dtype=np.int64)))
        offset += len(keep)
        if all(len(v) == per_cell for v in cells.values()):
            break
    else:
        short = {k: len(v) for k, v in cells.items() if len(v) < per_cell}
        raise RuntimeError(f"stratified sampling ran out of chunks, short cells: {short}")
    return Bodies.concat(parts)


def template() -> Bodies:
    """Neutral mean adult (all phenotypes 0.5, age 35, gender 0.5), no local controls, no pose."""
    model = get_model()
    labels = list(model.phenotype_labels)
    ph = np.full((1, len(labels)), 0.5, dtype=np.float32)
    mapping = _shape_distribution().morphological_age_mapping
    ph[0, labels.index("age")] = float(mapping.morphological_to_anny_age(torch.tensor([TEMPLATE_AGE]))[0])
    return build(ph, np.zeros((1, len(LOCAL_CONTROLS)), np.float32), np.zeros((1, 3), np.float32))


def bone_heads(body: Bodies, i: int = 0) -> dict[str, np.ndarray]:
    """Posed bone head positions for one body, in the same centred frame as its vertices.

    Anny's rest frame differs from its posed (reference A-pose) frame, so the heads are
    taken from the posed bone transforms and centred exactly like ``build`` centres vertices.
    """
    model = get_model()
    with torch.no_grad():
        out = model(
            pose_parameters=_pose_dict(body.pose[i : i + 1]),
            phenotype_kwargs=torch.from_numpy(body.phenotype[i : i + 1]),
            local_changes_kwargs=_local_tensor(body.local[i : i + 1]),
        )
    v = out["vertices"][0].numpy()
    heads = out["bone_poses"][0, :, :3, 3].numpy().copy()
    heads[:, 2] -= v[:, 2].min()
    heads[:, :2] -= heads[0, :2]
    return dict(zip(model.bone_labels, heads))


def describe(b: Bodies) -> str:
    """One-line-per-stratum summary used by scripts and tests."""
    lines = [f"n={len(b)}  height {b.height.min():.2f}-{b.height.max():.2f} m  mesh BMI {b.bmi.min():.1f}-{b.bmi.max():.1f}"]
    for name, labels, band in (("bmi", BMI_LABELS, b.bmi_band), ("sex", SEX_LABELS, b.sex), ("height", HEIGHT_LABELS, b.height_band)):
        counts = np.bincount(band, minlength=len(labels))
        lines.append(f"  {name}: " + ", ".join(f"{l}={c}" for l, c in zip(labels, counts)))
    return "\n".join(lines)

