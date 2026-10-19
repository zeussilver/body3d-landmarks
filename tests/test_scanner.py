import time

import numpy as np

from body3d import bodies, landmarks, scanner

SC = scanner.Scanner()
N = {n: i for i, n in enumerate(landmarks.NAMES)}


def test_rasteriser_matches_ray_casting(mixed, faces):
    v = mixed.vertices[-1]  # heaviest band
    depth, _ = SC.render(v, faces)
    for k in (0, 3):
        ref = scanner.ray_cast_depth(v, faces, SC, k)
        np.testing.assert_array_equal(np.isfinite(ref), np.isfinite(depth[k]))
        hit = np.isfinite(ref)
        assert np.abs(ref[hit] - depth[k][hit]).max() < 1e-6


def test_render_is_fast_enough(template, faces):
    SC.render(template.vertices[0], faces)
    t = time.perf_counter()
    for _ in range(3):
        SC.render(template.vertices[0], faces)
    assert (time.perf_counter() - t) / 3 < 0.1  # A1 target is 40 ms; loose bound against flaky timing


def test_backprojection_round_trip(template, faces):
    depth, _ = SC.render(template.vertices[0], faces)
    for k, pts in enumerate(SC.backproject(depth)):
        col, row, d = SC.to_pixels(SC.view_points(pts, k))
        np.testing.assert_allclose(col, np.rint(col), atol=1e-9)
        hit = np.isfinite(depth[k])
        np.testing.assert_allclose(np.sort(d), np.sort(depth[k][hit]), atol=1e-9)


def test_yaw_convention_front_and_back(template, faces, lm_idx, rng):
    s = scanner.scan(template.vertices[0], faces, template.vertices[0][lm_idx], rng, SC, noise=None)
    front, back = s["visible"][0], s["visible"][SC.n_views // 2]
    assert front[[N["bust_point_l"], N["bust_point_r"], N["waist_front"], N["knee_l"]]].all()
    assert back[[N["cervicale"], N["waist_back"], N["buttock_point"]]].all()
    assert not front[N["waist_back"]] and not back[N["waist_front"]]


def test_vertex_is_observable_from_the_silhouette(template, faces, lm_idx, rng):
    s = scanner.scan(template.vertices[0], faces, template.vertices[0][lm_idx], rng, SC, noise=None)
    assert s["observable"][:, N["vertex"]].all()
    assert (s["observable"] | ~s["visible"]).all()  # visible implies observable


def test_noise_is_bounded_deterministic_and_never_fills_holes(template, faces):
    clean, cos = SC.render(template.vertices[0], faces)
    a = scanner.add_noise(clean, cos, np.random.default_rng(3))
    b = scanner.add_noise(clean, cos, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    assert not (np.isfinite(a) & ~np.isfinite(clean)).any()
    both = np.isfinite(a)
    assert 0.5 < both.sum() / np.isfinite(clean).sum() < 1.0
    err = np.abs(a[both] - clean[both])
    assert np.percentile(err, 99) < 0.015
    np.testing.assert_allclose(a[both] * 1e3, np.round(a[both] * 1e3), atol=1e-6)


def test_sway_is_zero_mean_and_off_when_zero(template):
    v = template.vertices[0].astype(np.float64)
    aw = bodies.arm_weights()
    views = scanner.sway_views(v, SC.n_views, np.random.default_rng(1), scanner.Sway(), aw)
    assert np.linalg.norm(views.mean(0) - v, axis=1).max() < 5e-4  # ground truth = average position
    assert np.median(np.linalg.norm(views - v, axis=2)) > 2e-3  # but each view really moves
    still = scanner.sway_views(v, SC.n_views, np.random.default_rng(1), scanner.Sway(0, 0, 0, 0), aw)
    np.testing.assert_allclose(still, np.broadcast_to(v, still.shape), atol=1e-12)


def test_arm_drift_moves_arms_only_and_pivots_at_the_shoulder(template):
    v = template.vertices[0].astype(np.float64)
    aw = bodies.arm_weights()
    views = scanner.sway_views(v, SC.n_views, np.random.default_rng(2), scanner.Sway(0, 0, 0, 3.0), aw)
    moved = np.linalg.norm(views - v, axis=2).max(0)
    assert moved[aw.sum(1) == 0].max() < 1e-12 and moved[aw.max(1) > 0.99].max() > 0.01
    piv = scanner.arm_pivots(v, aw)
    assert np.linalg.norm(piv[0] - bodies.bone_heads(template)["upperarm01.L"]) < 0.03


def test_sway_leaves_ground_truth_unchanged(template, faces, lm_idx):
    v = template.vertices[0].astype(np.float64)
    a = scanner.scan(v, faces, v[lm_idx], np.random.default_rng(4), SC)
    b = scanner.scan(v, faces, v[lm_idx], np.random.default_rng(4), SC, sway=scanner.Sway(), arm_w=bodies.arm_weights())
    np.testing.assert_array_equal(a["landmarks"], b["landmarks"])
    assert not np.array_equal(np.isfinite(a["depth"]), np.isfinite(b["depth"]))
