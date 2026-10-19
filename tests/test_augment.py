import numpy as np
import pytest
import trimesh

from body3d import augment, bodies


@pytest.fixture(scope="module")
def inflate(faces):
    return augment.Inflate(faces, augment.region_weights(), p_apply=1.0)


def test_region_weights_cover_trunk_not_head(inflate, template, lm_idx):
    w = inflate.weights
    assert w.shape[1] == len(augment.REGIONS) and (w >= 0).all() and (w.sum(1) <= 1 + 1e-5).all()
    vertex = lm_idx[0]  # top of the head: no region, never inflated
    assert w[vertex].sum() == 0


def test_inflation_moves_vertices_along_normals_and_is_seeded(inflate, template, faces):
    v = template.vertices[0].astype(np.float64)
    a = inflate(v, np.random.default_rng(3))
    b = inflate(v, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    d = a - v
    n = augment.vertex_normals(v, faces)
    moved = np.linalg.norm(d, axis=1) > 1e-9
    np.testing.assert_allclose(np.abs((d[moved] * n[moved]).sum(1)), np.linalg.norm(d[moved], axis=1), rtol=1e-6)


def test_offsets_stay_in_range_and_inflation_adds_volume_on_average(inflate, template, faces):
    rng = np.random.default_rng(0)
    offs = np.array([inflate.offsets(rng) for _ in range(200)])
    assert (offs >= inflate.lo - 1e-12).all() and (offs <= inflate.hi + 1e-12).all()
    v = template.vertices[0].astype(np.float64)
    base = trimesh.Trimesh(v, faces, process=False).volume
    vols = [trimesh.Trimesh(inflate(v, rng), faces, process=False).volume for _ in range(10)]
    assert np.mean(vols) > base * 1.1


def test_extrapolation_keeps_identity_and_mostly_adds_weight(mixed):
    ph, lc = augment.extrapolate(mixed.phenotype, mixed.local, np.random.default_rng(0))
    labels = list(bodies.get_model().phenotype_labels)
    keep = [labels.index(k) for k in ("gender", "age", "muscle", "height", "proportions")]
    np.testing.assert_array_equal(ph[:, keep], mixed.phenotype[:, keep])
    wi = labels.index("weight")
    assert (ph[:, wi] != mixed.phenotype[:, wi]).mean() > 0.5
    untouched = [i for i, n in enumerate(bodies.LOCAL_NAMES) if n not in augment.ADIPOSITY_LOCALS]
    np.testing.assert_array_equal(lc[:, untouched], mixed.local[:, untouched])
