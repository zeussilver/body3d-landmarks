"""Train the multi-view landmark model and score it on the fixed test set (SYNTHETIC data).

    python -m body3d.train --cond e0 --seed 0    # full range        -> results/e0_s0.json
    python -m body3d.train --cond e1 --seed 0    # BMI 18.5-27 only
    python -m body3d.train --cond e2 --seed 0    # e1 + normal-offset inflation
    python -m body3d.train --cond e3 --seed 0    # e1 + shape-space extrapolation

Training scans are rendered on the fly (new placement, sway and sensor noise every epoch).
Model selection uses the validation split of the same distribution; the test split is
scored once, with the best-validation checkpoint.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset, Subset  # noqa: E402

from body3d import bodies, data, evaluate, landmarks  # noqa: E402
from body3d.model import LandmarkModel, depth_to_input, loss_fn  # noqa: E402
from body3d.scanner import scan  # noqa: E402

RUNS = data.ROOT / "runs"
EPOCHS = 10
# Experimental conditions (PLAN 2.7). All train on 8,000 subjects and validate on the
# validation split of their own training distribution.
CONDITIONS = {
    "e0": {"dist": "full"},
    "e1": {"dist": "narrow"},
    "e2": {"dist": "narrow", "augment": "inflate"},
    "e3": {"dist": "narrow", "alt": "extrap"},
}


def sample_tensors(depth: np.ndarray, lm: np.ndarray):
    col, row, _ = data.SCANNER.project(np.asarray(lm, np.float64))
    px = np.stack([col, row], -1)
    return depth_to_input(depth), torch.tensor(np.array(lm), dtype=torch.float32), torch.as_tensor(px, dtype=torch.float32)


class TrainScans(Dataset):
    """Meshes from ``data.train_meshes``; every access re-renders the scan (placement, sway, noise).

    The random draws depend on (seed, worker seed, index). Workers persist across epochs, so a body
    cycles through at most ``workers`` perturbations instead of a new one every epoch.
    """

    def __init__(self, dist: str, faces: np.ndarray, idx: np.ndarray, arm_w: np.ndarray, augment=None,
                 seed: int = 0, alt_dir=None, p_alt: float = 0.5):
        self.dir = data.train_meshes(dist)
        self.n = len(np.load(self.dir / "meta.npz")["bmi"])
        self.faces, self.idx, self.arm_w, self.augment, self.seed = faces, idx, arm_w, augment, seed
        self.alt_dir, self.p_alt = alt_dir, p_alt  # E3: per-subject variant meshes, same index order
        self.v = self.alt = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.v is None:  # open lazily so each worker memory-maps instead of receiving a copy
            self.v = np.load(self.dir / "vertices.npy", mmap_mode="r")
            if self.alt_dir is not None:
                self.alt = np.load(self.alt_dir / "vertices.npy", mmap_mode="r")
        rng = np.random.default_rng([self.seed, torch.initial_seed() % 2**32, i])
        src = self.alt if self.alt is not None and rng.random() < self.p_alt else self.v
        v = np.asarray(src[i], dtype=np.float64)
        if self.augment is not None:
            v = self.augment(v, rng)
        s = scan(v, self.faces, v[self.idx], rng, data.SCANNER, data.NOISE, data.SWAY, self.arm_w)
        return sample_tensors(s["depth"], s["landmarks"])


class FixedScans(Dataset):
    """Cached scans (validation / test)."""

    def __init__(self, scans: dict):
        self.depth_mm, self.lm = scans["depth_mm"], scans["landmarks"]

    def __len__(self):
        return len(self.lm)

    def __getitem__(self, i):
        return sample_tensors(data.decode_depth(np.asarray(self.depth_mm[i])), self.lm[i])


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def predict(model: LandmarkModel, scans: dict, batch: int = 16, workers: int = 4) -> np.ndarray:
    """(N, K, 3) predicted landmarks in metres for cached scans."""
    model.eval()
    dev = next(model.parameters()).device
    dl = DataLoader(FixedScans(scans), batch, num_workers=workers)
    return np.concatenate([model(x.to(dev))["points"].cpu().numpy() for x, _, _ in dl])


def train(run: str, dist: str, epochs: int, seed: int = 0, batch: int = 8, lr: float = 2e-3,
          augment: str | None = None, alt: str | None = None, workers: int = 8, limit: int | None = None,
          meta: dict | None = None) -> dict:
    torch.manual_seed(seed)
    out_dir = RUNS / run
    out_dir.mkdir(parents=True, exist_ok=True)
    faces, idx = bodies.faces(), landmarks.load()
    aug = None
    if augment == "inflate":
        from body3d.augment import Inflate, region_weights

        aug = Inflate(faces, region_weights())
    alt_dir = data.train_meshes_extrapolated(dist) if alt == "extrap" else None
    ds = TrainScans(dist, faces, idx, bodies.arm_weights(), aug, seed, alt_dir)
    if limit:
        ds = Subset(ds, range(limit))
    dl = DataLoader(ds, batch, shuffle=True, num_workers=workers, persistent_workers=True, drop_last=True,
                    prefetch_factor=4, generator=torch.Generator().manual_seed(seed))
    _, val = data.val_split(dist)
    if limit:
        val = {k: v[: max(100, limit // 8)] for k, v in val.items()}

    dev = device()
    model = LandmarkModel().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl), pct_start=0.1)
    best, best_epoch, t0 = np.inf, -1, time.perf_counter()
    log = (out_dir / "log.jsonl").open("w")
    for ep in range(epochs):
        model.train()
        sums, n = {}, 0
        for x, p, px in dl:
            x, p, px = x.to(dev, non_blocking=True), p.to(dev), px.to(dev)
            loss, parts = loss_fn(model(x), p, px)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        val_mm = float(evaluate.errors_mm(predict(model, val), val["landmarks"]).mean())
        rec = {"epoch": ep, "val_mm": val_mm, "minutes": (time.perf_counter() - t0) / 60,
               **{f"train_{k}": v / n for k, v in sums.items()}}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
        if val_mm < best:
            best, best_epoch = val_mm, ep
            torch.save(model.state_dict(), out_dir / "best.pt")
    train_minutes = (time.perf_counter() - t0) / 60

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=dev))
    test_bodies, test = data.test_split()
    pred = predict(model, test)
    np.save(out_dir / "test_pred.npy", pred)
    info = {"model": "per-view U-Net (heatmaps + confidence) + weighted LS triangulation",
            "train_distribution": dist, "train_bmi_range": data.DISTRIBUTIONS[dist]["bmi_range"],
            "train_bodies": len(ds), "epochs": epochs, "seed": seed, "batch_bodies": batch, "lr": lr,
            "augment": augment, "alt_meshes": alt,
            "best_epoch": best_epoch, "best_val_mm": best, "train_minutes": round(train_minutes, 1),
            "params": sum(p.numel() for p in model.parameters()), **(meta or {})}
    return evaluate.evaluate(run, pred, test["landmarks"], test_bodies, info,
                             results_dir=None if not limit else out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=list(CONDITIONS), help="experimental condition; sets --dist and fixes")
    ap.add_argument("--run", help="run name (default: <cond>_s<seed>)")
    ap.add_argument("--dist", choices=list(data.DISTRIBUTIONS))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N training bodies, results stay in runs/")
    a = ap.parse_args()
    cond = CONDITIONS.get(a.cond, {})
    dist = cond.get("dist", a.dist)
    if dist is None:
        ap.error("give --cond or --dist")
    run = a.run or f"{a.cond}_s{a.seed}"
    res = train(run, dist, a.epochs, a.seed, a.batch, a.lr, augment=cond.get("augment"), alt=cond.get("alt"),
                workers=a.workers, limit=a.limit, meta={"condition": a.cond} if a.cond else None)
    print(evaluate.band_table([res]))


if __name__ == "__main__":
    main()
