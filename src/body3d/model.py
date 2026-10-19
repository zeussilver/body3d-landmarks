"""Multi-view landmark model: a shared per-view U-Net + differentiable triangulation.

Each depth view gives K heatmaps (half resolution) and K confidence weights. Soft-argmax
turns heatmaps into sub-pixel image positions. In an orthographic turntable view at yaw
theta a point (x, y, z) lands at u = x cos(theta) - y sin(theta), w = z (see
``scanner``), so the 3D landmark is a closed-form weighted least-squares solve over views.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from body3d.scanner import Scanner

DEPTH_SCALE = 0.25  # metres: depth is fed as (d - per-view median) / DEPTH_SCALE


def block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class ViewNet(nn.Module):
    """(B, 2, H, W) depth + mask -> heatmap logits (B, K, H/2, W/2) and confidence logits (B, K)."""

    def __init__(self, k: int = 16, ch: tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(2, 16, 3, 1, 1), nn.ReLU(inplace=True), block(16, ch[0], 2))  # 1/2
        self.down = nn.ModuleList(block(a, b, 2) for a, b in zip(ch[:-1], ch[1:]))  # 1/4, 1/8, 1/16
        self.up = nn.ModuleList(block(b + a, a) for a, b in zip(ch[-2::-1], ch[:0:-1]))
        self.heat = nn.Conv2d(ch[0], k, 1)
        self.conf = nn.Linear(ch[-1], k)

    def forward(self, x):
        skips = [self.stem(x)]
        for d in self.down:
            skips.append(d(skips[-1]))
        y = skips.pop()
        conf = self.conf(y.mean((2, 3)))
        for u in self.up:
            s = skips.pop()
            y = u(torch.cat([F.interpolate(y, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        return self.heat(y), conf


def soft_argmax(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, K, h, w) -> expected (col, row) in heatmap cells (B, K, 2) and the softmax maps."""
    b, k, h, w = logits.shape
    p = logits.flatten(2).softmax(-1).view(b, k, h, w)
    cols = torch.arange(w, device=logits.device, dtype=p.dtype)
    rows = torch.arange(h, device=logits.device, dtype=p.dtype)
    return torch.stack([(p.sum(2) * cols).sum(-1), (p.sum(3) * rows).sum(-1)], -1), p


class Triangulator(nn.Module):
    """Image positions (B, V, K, 2) in full-resolution pixels + weights (B, V, K) -> (B, K, 3) metres."""

    def __init__(self, scanner: Scanner = Scanner(), ridge: float = 1e-6):
        super().__init__()
        th = torch.deg2rad(torch.tensor(scanner.yaws, dtype=torch.float32))
        self.register_buffer("c", torch.cos(th).view(1, -1, 1))
        self.register_buffer("s", torch.sin(th).view(1, -1, 1))
        self.pixel, self.w, self.h, self.ridge = scanner.pixel, scanner.width, scanner.height, ridge

    def to_metric(self, px: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = (px[..., 0] + 0.5) * self.pixel - self.w * self.pixel / 2
        z = self.h * self.pixel - (px[..., 1] + 0.5) * self.pixel
        return u, z

    def forward(self, px: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        u, z = self.to_metric(px)
        c, s, a = self.c, self.s, weight
        # minimise sum_v a (c x - s y - u)^2 -> 2x2 normal equations per landmark
        m11, m12, m22 = (a * c * c).sum(1) + self.ridge, -(a * c * s).sum(1), (a * s * s).sum(1) + self.ridge
        b1, b2 = (a * c * u).sum(1), -(a * s * u).sum(1)
        det = m11 * m22 - m12 * m12
        x, y = (m22 * b1 - m12 * b2) / det, (m11 * b2 - m12 * b1) / det
        zz = (a * z).sum(1) / a.sum(1)
        return torch.stack([x, y, zz], -1)


class LandmarkModel(nn.Module):
    def __init__(self, k: int = 16, scanner: Scanner = Scanner()):
        super().__init__()
        self.net = ViewNet(k)
        self.tri = Triangulator(scanner)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (B, V, 2, H, W) -> 3D landmarks (B, K, 3), per-view pixels (B, V, K, 2), heatmaps, weights."""
        b, v = x.shape[:2]
        logits, conf = self.net(x.flatten(0, 1))
        cells, prob = soft_argmax(logits.float())
        px = (cells + 0.5) * 2 - 0.5  # half-resolution cells -> full-resolution pixel coordinates
        weight = F.softplus(conf.float()) + 1e-3
        px, weight = px.view(b, v, *px.shape[1:]), weight.view(b, v, -1)
        return {"points": self.tri(px, weight), "px": px, "weight": weight, "prob": prob}


def depth_to_input(depth: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Depth (..., V, H, W) metres with inf = no return -> network input (..., V, 2, H, W)."""
    d = torch.as_tensor(depth, dtype=torch.float32)
    mask = torch.isfinite(d)
    filled = torch.where(mask, d, torch.nan)
    med = filled.flatten(-2).nanmedian(-1).values[..., None, None]
    rel = torch.where(mask, (d - med) / DEPTH_SCALE, torch.zeros_like(d))
    return torch.stack([rel, mask.float()], -3)


def gaussian_targets(px: torch.Tensor, h: int, w: int, sigma: float = 1.0) -> torch.Tensor:
    """Full-res pixel targets (..., K, 2) -> normalised Gaussian maps on the half-res grid (..., K, h, w)."""
    cell = (px + 0.5) / 2 - 0.5
    cx, cy = cell[..., 0, None, None], cell[..., 1, None, None]  # (..., K, 1, 1)
    cols = torch.arange(w, device=px.device, dtype=px.dtype)
    rows = torch.arange(h, device=px.device, dtype=px.dtype)[:, None]
    g = torch.exp(-((cols - cx) ** 2 + (rows - cy) ** 2) / (2 * sigma**2))
    return g / g.flatten(-2).sum(-1)[..., None, None].clamp_min(1e-12)


def loss_fn(out: dict, gt_points: torch.Tensor, gt_px: torch.Tensor, w3d: float = 0.1, wheat: float = 0.1):
    """3D Euclidean error (mm, scaled) + 2D L1 (half-res cells) + heatmap cross-entropy."""
    err3d = ((out["points"] - gt_points) ** 2).sum(-1).add(1e-8).sqrt() * 1e3
    l2d = (out["px"] - gt_px).abs().sum(-1).mean() / 2
    prob = out["prob"]
    target = gaussian_targets(gt_px.flatten(0, 1), *prob.shape[-2:])
    lheat = -(target * prob.clamp_min(1e-9).log()).flatten(-2).sum(-1).mean()
    total = w3d * err3d.mean() + l2d + wheat * lheat
    return total, {"mm": err3d.mean().item(), "px2d": l2d.item() * 2, "heat": lheat.item()}
