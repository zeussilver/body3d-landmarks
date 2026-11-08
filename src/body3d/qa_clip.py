"""Open vision-language foundation model (CLIP ViT-B/32, LAION-2B) on the scan check.

Two ways to use it:

- zero-shot: compare the scan image with a text description of each class, no labels;
- linear probe: freeze CLIP, embed each of the 8 views, and train a logistic regression on
  k labelled dev scans per class (learning curve k = 5 ... 100).

    uv sync --group qa && python -m body3d.qa_clip
"""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTPS download; the Xet transfer path can stall

import numpy as np  # noqa: E402
import torch  # noqa: E402

from body3d import qa  # noqa: E402
from body3d.data import RESULTS, decode_depth  # noqa: E402

MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
HF_REPO = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
PROMPTS = {  # several phrasings per class; class score = mean over phrasings
    "ok": ["eight depth images of a person standing still with arms held away from the body",
           "a complete body scan of a person in an A-pose, every view clear"],
    "arms_down": ["eight depth images of a person standing with arms pressed against the sides of the body",
                  "a body scan where the arms hang down touching the torso"],
    "moved": ["eight depth images where the person turned or moved between views",
              "a body scan where some views show the person facing a different direction than expected"],
    "cropped": ["eight depth images where part of the person is cut off at the edge of the image",
                "a body scan where the person stands off-centre and leaves the frame"],
    "dropout": ["eight depth images where one view is blank or has a large missing region",
                "a body scan with a black empty panel or a big hole in the body"],
}
SHOTS = (5, 10, 20, 50, 100)
DRAWS = 5


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model():
    """CLIP from the safetensors weights (loading them runs no code, unlike a pickled .bin)."""
    import open_clip
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(HF_REPO, "open_clip_model.safetensors")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=weights)
    return model.eval().to(device()), preprocess, open_clip.get_tokenizer(MODEL)


def _square(im):
    from PIL import Image

    side = max(im.size)
    out = Image.new("RGB", (side, side))
    out.paste(im, ((side - im.size[0]) // 2, (side - im.size[1]) // 2))
    return out


@torch.no_grad()
def embed(split: str, model, preprocess) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-view embeddings (N, 8, 512) and whole-grid embeddings (N, 512), L2-normalised,
    cached under data/qa/<split>/ with the time it took (milliseconds per scan, 9 images)."""
    d = qa.QA_DIR / split
    pv, pg, pt = d / "clip_views.npy", d / "clip_grid.npy", d / "clip_embed_ms.txt"
    if pv.exists() and pg.exists() and pt.exists():
        return np.load(pv), np.load(pg), float(pt.read_text())
    s, dev = qa.load_split(split), device()
    views, grids = [], []
    t0 = time.perf_counter()
    for depth_mm in s["depth_mm"]:
        depth = decode_depth(np.asarray(depth_mm))
        x = torch.stack([preprocess(im) for im in qa.view_images(depth)] + [preprocess(_square(qa.grid_image(depth)))])
        e = model.encode_image(x.to(dev)).float()
        e = (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()
        views.append(e[:-1])
        grids.append(e[-1])
    ms = (time.perf_counter() - t0) / len(views) * 1e3
    views, grids = np.stack(views), np.stack(grids)
    np.save(pv, views)
    np.save(pg, grids)
    pt.write_text(f"{ms:.3f}\n")
    return views, grids, ms


@torch.no_grad()
def zero_shot(grids: np.ndarray, model, tokenizer) -> np.ndarray:
    dev = device()
    scores = []
    for c in qa.CLASSES:
        t = model.encode_text(tokenizer(PROMPTS[c]).to(dev)).float()
        t = (t / t.norm(dim=-1, keepdim=True)).cpu().numpy()
        scores.append((grids @ t.T).mean(1))
    return np.argmax(np.stack(scores, 1), 1)


def probe(c: float = 1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))


def choose_c(x: np.ndarray, y: np.ndarray) -> float:
    """Regularisation by 5-fold cross-validation on dev only."""
    from sklearn.model_selection import cross_val_score

    grid = (0.001, 0.01, 0.1, 1.0)
    scores = [cross_val_score(probe(c), x, y, cv=5, scoring="f1_macro").mean() for c in grid]
    return grid[int(np.argmax(scores))]


def main() -> None:
    model, preprocess, tokenizer = load_model()
    vd, gd, _ = embed("dev", model, preprocess)
    vt, gt, ms = embed("test", model, preprocess)
    yd, yt = qa.load_split("dev")["labels"], qa.load_split("test")["labels"]
    xd, xt = vd.reshape(len(vd), -1), vt.reshape(len(vt), -1)

    zs = zero_shot(gt, model, tokenizer)
    np.save(qa.QA_DIR / "test" / "pred_clip_zero_shot.npy", zs)
    c = choose_c(xd, yd)
    full = probe(c).fit(xd, yd).predict(xt)
    np.save(qa.QA_DIR / "test" / "pred_clip_probe.npy", full)
    rng = np.random.default_rng(0)
    curve = {}
    for k in SHOTS:
        accs, f1s = [], []
        for _ in range(DRAWS if k < 100 else 1):
            idx = np.concatenate([rng.choice(np.flatnonzero(yd == cl), k, replace=False) for cl in range(len(qa.CLASSES))])
            m = qa.metrics(yt, probe(c).fit(xd[idx], yd[idx]).predict(xt), n_boot=1)
            accs.append(m["accuracy"])
            f1s.append(m["macro_f1"])
        curve[k] = {"accuracy_mean": float(np.mean(accs)), "accuracy_sd": float(np.std(accs)),
                    "macro_f1_mean": float(np.mean(f1s)), "draws": len(accs)}
    res = {"model": f"open_clip {MODEL} / {PRETRAINED} (frozen)", "device": str(device()),
           "embed_ms_per_scan": round(ms, 1), "probe_C": c,
           "zero_shot_test": qa.metrics(yt, zs), "probe_full_dev_test": qa.metrics(yt, full),
           "probe_learning_curve_test": curve, "prompts": PROMPTS}
    (RESULTS / "scan_qa_clip.json").write_text(json.dumps(res, indent=1) + "\n")
    print("zero-shot  ", qa.summary_line(res["zero_shot_test"]))
    print("probe (100)", qa.summary_line(res["probe_full_dev_test"]))
    for k, v in curve.items():
        print(f"  k={k:3d}/class  acc {v['accuracy_mean']:.3f} +- {v['accuracy_sd']:.3f}")
    print("embed ms/scan", res["embed_ms_per_scan"], res["device"], "C", c)


if __name__ == "__main__":
    main()
