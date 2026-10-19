import numpy as np

from body3d import baseline, scanner
from body3d.scanner import rot_z


def test_similarity_yaw_recovers_known_transform(rng):
    p = rng.normal(size=(200, 3))
    r, s, t = rot_z(23.0), 1.07, np.array([0.02, -0.01, 0.005])
    s2, r2, t2 = baseline.similarity_yaw(p, s * p @ r.T + t)
    assert abs(s2 - s) < 1e-9 and np.allclose(r2, r) and np.allclose(t2, t)


def test_icp_recovers_placement_of_the_template(template, faces):
    v = template.vertices[0]
    pts = v[baseline.template_points(v, faces)]
    r, s, t = rot_z(12.0), 0.93, np.array([0.015, -0.02, 0.0])
    target = s * v @ r.T + t
    s2, r2, t2 = baseline.icp_similarity(pts, target)
    err = np.linalg.norm(baseline.apply(s2, r2, t2, v) - target, axis=1)
    assert err.mean() < 1e-3


def test_b0_on_a_clean_template_scan_is_close(template, faces, lm_idx, rng):
    v = template.vertices[0]
    s = scanner.scan(v, faces, v[lm_idx], rng, noise=None)
    snapped, _ = baseline.predict(s["depth"].astype(np.float64), v[baseline.template_points(v, faces)], v[lm_idx])
    err = np.linalg.norm(snapped - s["landmarks"], axis=1) * 1e3
    assert np.median(err) < 10.0
