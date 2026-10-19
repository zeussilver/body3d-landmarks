import numpy as np
import torch

from body3d import scanner
from body3d.model import LandmarkModel, Triangulator, depth_to_input, gaussian_targets, loss_fn, soft_argmax

SC = scanner.Scanner()


def _pixels(points: np.ndarray) -> torch.Tensor:
    col, row, _ = SC.project(points)  # (V, K)
    return torch.tensor(np.stack([col, row], -1), dtype=torch.float32)[None]


def test_triangulation_is_exact_for_consistent_views(rng):
    p = rng.uniform([-0.3, -0.2, 0.1], [0.3, 0.2, 1.9], (16, 3))
    w = torch.rand(1, SC.n_views, 16) + 0.1
    rec = Triangulator(SC)(_pixels(p), w)[0].numpy()
    assert np.abs(rec - p).max() < 1e-5


def test_low_weight_suppresses_a_bad_view(rng):
    p = rng.uniform(-0.2, 0.2, (16, 3)) + [0, 0, 1.0]
    px = _pixels(p)
    px[0, 2] += 15.0  # one view off by 15 px (about 13 cm)
    tri = Triangulator(SC)
    equal = tri(px, torch.ones(1, SC.n_views, 16))[0].numpy()
    w = torch.ones(1, SC.n_views, 16)
    w[0, 2] = 1e-3
    down = tri(px, w)[0].numpy()
    assert np.abs(down - p).max() < 1e-3 < np.abs(equal - p).max()


def test_soft_argmax_and_gaussian_targets_agree():
    px = torch.tensor([[[50.0, 101.0], [10.0, 20.0]]])  # full-resolution pixels
    target = gaussian_targets(px, 128, 80, sigma=0.5)
    assert torch.allclose(target.flatten(-2).sum(-1), torch.ones(1, 2))
    cells, _ = soft_argmax(target.clamp_min(1e-12).log())
    np.testing.assert_allclose(((cells + 0.5) * 2 - 0.5).numpy(), px.numpy(), atol=0.05)


def test_depth_to_input_masks_background():
    d = np.full((2, 4, 4), np.inf, dtype=np.float32)
    d[0, 1:3, 1:3] = [[1.0, 1.1], [1.2, 1.3]]
    x = depth_to_input(d)
    assert x.shape == (2, 2, 4, 4)
    assert torch.equal(x[:, 1].bool(), torch.from_numpy(np.isfinite(d)))
    assert (x[1] == 0).all() and (x[0, 0][~torch.from_numpy(np.isfinite(d[0]))] == 0).all()


def test_model_forward_backward_shapes():
    torch.manual_seed(0)
    m = LandmarkModel(scanner=SC)
    x = torch.randn(1, SC.n_views, 2, SC.height, SC.width)
    out = m(x)
    assert out["points"].shape == (1, 16, 3) and out["px"].shape == (1, SC.n_views, 16, 2)
    assert (out["weight"] > 0).all()
    loss, parts = loss_fn(out, torch.zeros(1, 16, 3), torch.full((1, SC.n_views, 16, 2), 50.0))
    loss.backward()
    assert np.isfinite(loss.item()) and all(np.isfinite(v) for v in parts.values())
