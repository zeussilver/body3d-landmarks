import numpy as np
import trimesh

from body3d import bodies


def test_sampling_is_reproducible():
    a, b, c = (bodies.sample_bodies(4, seed=s, chunk=8) for s in (7, 7, 8))
    np.testing.assert_array_equal(a.vertices, b.vertices)
    assert not np.allclose(a.vertices, c.vertices)


def test_frame_units_and_root(mixed):
    assert np.allclose(mixed.vertices[..., 2].min(1), 0.0)
    assert ((mixed.height > 1.3) & (mixed.height < 2.2)).all()
    root = bodies.bone_heads(mixed, 0)["root"]
    assert np.abs(root[:2]).max() < 1e-5


def test_implausible_bodies_are_rejected(mixed):
    assert (mixed.bmi >= bodies.BMI_PLAUSIBLE[0]).all() and (mixed.bmi < bodies.BMI_PLAUSIBLE[1]).all()


def test_bmi_range_filter():
    b = bodies.sample_bodies(20, seed=9, bmi_range=(18.5, 27.0), chunk=64)
    assert len(b) == 20 and (b.bmi >= 18.5).all() and (b.bmi < 27.0).all()


def test_stratified_counts(mixed):
    assert np.bincount(mixed.bmi_band, minlength=5).tolist() == [2] * 5
    assert np.bincount(mixed.sex).tolist() == [5, 5]


def test_mesh_bmi_matches_independent_volume(template, faces):
    # template has no pose jitter, so posed and rest volumes agree closely
    vol = trimesh.Trimesh(template.vertices[0], faces, process=False).volume
    assert abs(vol * 980 / template.height[0] ** 2 - template.bmi[0]) / template.bmi[0] < 0.01


def test_save_load_roundtrip(mixed, tmp_path):
    mixed.save(tmp_path / "b.npz")
    back = bodies.Bodies.load(tmp_path / "b.npz")
    np.testing.assert_array_equal(back.vertices, mixed.vertices)
    np.testing.assert_array_equal(back.bmi_band, mixed.bmi_band)
