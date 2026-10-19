import numpy as np
import pytest

from body3d import augment, bodies, change, stl
from body3d.scanner import rot_x, rot_z


def cylinder(rx: float, ry: float, n: int = 128, h: float = 1.0):
    """Closed elliptic cylinder mesh, z from 0 to h, outward winding."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[rx * np.cos(a), ry * np.sin(a)]
    v = np.r_[np.c_[ring, np.zeros(n)], np.c_[ring, np.full(n, h)], [[0, 0, 0], [0, 0, h]]]
    i, j = np.arange(n), (np.arange(n) + 1) % n
    side = np.r_[np.c_[i, j, j + n], np.c_[i, j + n, i + n]]
    caps = np.r_[np.c_[np.full(n, 2 * n), j, i], np.c_[np.full(n, 2 * n + 1), i + n, j + n]]
    return v, np.r_[side, caps]


def test_stl_binary_and_ascii_round_trip(tmp_path, template, faces):
    v = template.vertices[0].astype(np.float64) * 1e3
    for ascii in (False, True):
        path = tmp_path / f"b{ascii}.stl"
        stl.write(path, v[:200], faces[:100] % 200, "SYNTHETIC", ascii=ascii)
        tris = stl.read_triangles(path)
        np.testing.assert_allclose(tris, v[:200][faces[:100] % 200], atol=1e-3)
    vw, fw = stl.read(tmp_path / "bFalse.stl")
    np.testing.assert_allclose(vw[fw], v[:200][faces[:100] % 200], atol=1e-3)


def test_stl_rejects_garbage(tmp_path):
    (tmp_path / "bad.stl").write_bytes(b"\x00" * 90)
    with pytest.raises(ValueError):
        stl.read_triangles(tmp_path / "bad.stl")


def test_slicer_matches_analytic_perimeters():
    for rx, ry in ((0.15, 0.15), (0.18, 0.12)):
        v, f = cylinder(rx, ry)
        (loop,) = change.slice_loops(v, f, 0.37)
        contour, tape = change.perimeters(loop)
        h = ((rx - ry) / (rx + ry)) ** 2
        exact = np.pi * (rx + ry) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))  # Ramanujan
        assert abs(contour - exact) / exact < 0.005 and abs(tape - contour) < 1e-9


def test_slicer_finds_separate_loops_and_tape_spans_concavity(template, faces):
    v = template.vertices[0].astype(np.float64)
    assert len(change.slice_loops(v, faces, 0.3 * v[:, 2].max())) == 2  # two shins
    loops = change.slice_loops(v, faces, 0.64 * v[:, 2].max())  # waist height: trunk + 2 arms
    trunk = max(loops, key=change.polygon_area)
    contour, tape = change.perimeters(trunk)
    assert len(loops) >= 3 and tape <= contour + 1e-9


def test_icp_recovers_a_known_rigid_transform(template, faces):
    v = template.vertices[0].astype(np.float64)
    r, t = rot_z(14.0) @ rot_x(2.0), np.array([0.04, -0.03, 0.01])
    moved = v @ r.T + t
    r2, t2, _ = change.icp(moved, v, augment.vertex_normals(v, faces))
    rms = np.sqrt(((moved @ r2.T + t2 - v) ** 2).sum(1).mean()) * 1e3
    assert rms < 2.0


def test_topology_and_surface_paths_agree(template, faces):
    v0 = template.vertices[0].astype(np.float64)
    inflate = augment.Inflate(faces, augment.region_weights(), p_apply=1.0)
    vt = inflate(v0, np.random.default_rng(5)) @ rot_z(10.0).T + [0.02, 0.01, 0.0]
    aw = bodies.arm_weights()
    _, disp_a = change.topology_change(v0, vt, faces, aw)
    _, disp_b, _ = change.surface_change(v0, faces, vt, faces)
    ok = ~np.isnan(disp_a)
    assert np.median(np.abs(disp_a[ok] - disp_b[ok])) * 1e3 < 3.0
    assert np.nanpercentile(disp_a, 90) > 0.005  # inflated trunk shows up as gain (head is never inflated)


def test_loss_curve_shape():
    vals = [change.loss_curve(w) for w in change.WEEKS]
    assert vals[0] == 0 and vals[-1] == pytest.approx(1) and np.all(np.diff(vals) > 0)
    assert vals[1] > 8 / 24  # fast early loss, then plateau
