import numpy as np

from body3d import bodies, landmarks

N = {n: i for i, n in enumerate(landmarks.NAMES)}


def test_frozen_json_matches_rules(template, lm_idx):
    sel = landmarks.select(template.vertices[0], bodies.bone_heads(template))
    assert [sel[n] for n in landmarks.NAMES] == lm_idx.tolist()


def test_left_right_pairs_mirror_within_10mm(template):
    sel = dict(zip(landmarks.NAMES, landmarks.load().tolist()))
    assert max(landmarks.mirror_residuals_mm(template.vertices[0], sel).values()) < 10.0


def test_anatomical_order_holds_on_every_body_type(template, mixed, lm_idx):
    chain = ["vertex", "cervicale", "acromion_l", "bust_point_l", "waist_side_l", "crotch", "knee_l", "ankle_l"]
    for v in np.concatenate([template.vertices, mixed.vertices]):
        p = v[lm_idx]
        z = [p[N[n], 2] for n in chain]
        assert all(a > b for a, b in zip(z, z[1:])), z
        assert p[N["waist_front"], 1] < p[N["waist_back"], 1]  # body faces -y
        for name in landmarks.NAMES:
            if name.endswith("_l"):
                assert p[N[name], 0] > 0 > p[N[name[:-2] + "_r"], 0]
