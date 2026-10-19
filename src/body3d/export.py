"""Export the per-view network to ONNX and write Python-vs-C# parity fixtures (SYNTHETIC).

    python -m body3d.export     # -> dotnet/fixtures/{viewnet.onnx, *.stl, *.depth.f32, expected.json}

The C# service renders, normalises and triangulates itself; only the CNN runs in
ONNX Runtime. Fixtures use clean renders (no sensor noise, no sway) so both sides
compute the same thing deterministically.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import torch

from body3d import change, landmarks, stl
from body3d.data import ROOT, SCANNER
from body3d.model import LandmarkModel, depth_to_input

FIXTURES = ROOT / "dotnet" / "fixtures"
CHECKPOINT = ROOT / "runs" / "e0_s0" / "best.pt"
SCANS = ("s1/week00", "s3/week08", "s4/week24")  # from data/trajectories (python -m body3d.change)
TOLERANCES = {"depth_mm": 1.0, "depth_pixel_fraction": 0.999, "landmark_mm": 1.0, "girth_mm": 0.1}


def load_model() -> LandmarkModel:
    m = LandmarkModel()
    m.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    return m.eval()


def export_onnx(model: LandmarkModel, path) -> None:
    x = torch.zeros(2, 2, SCANNER.height, SCANNER.width)
    torch.onnx.export(model.net, (x,), str(path), input_names=["depth"], output_names=["heatmap", "confidence"],
                      dynamic_axes={"depth": {0: "views"}, "heatmap": {0: "views"}, "confidence": {0: "views"}},
                      opset_version=17, dynamo=False)


def main() -> None:
    import onnxruntime as ort

    FIXTURES.mkdir(parents=True, exist_ok=True)
    model = load_model()
    onnx_path = FIXTURES / "viewnet.onnx"
    export_onnx(model, onnx_path)
    sess = ort.InferenceSession(str(onnx_path))

    entries, onnx_diff = [], []
    for name in SCANS:
        src = ROOT / "data" / "trajectories" / f"{name}.stl"
        tag = name.replace("/", "_")
        shutil.copy(src, FIXTURES / f"{tag}.stl")
        v_mm, f = stl.read(src)
        v = v_mm / 1e3
        depth, _ = SCANNER.render(v, f)
        depth.astype("<f4").tofile(FIXTURES / f"{tag}.depth.f32")
        x = depth_to_input(depth)  # (V, 2, H, W)
        with torch.no_grad():
            out = model(x[None])
            heat, conf = model.net(x)
        h_onnx, c_onnx = sess.run(None, {"depth": x.numpy()})
        onnx_diff.append(float(max(np.abs(h_onnx - heat.numpy()).max(), np.abs(c_onnx - conf.numpy()).max())))
        lm = out["points"][0].numpy().astype(np.float64)
        g = change.girths(v, f, lm)
        entries.append({"stl": f"{tag}.stl", "depth": f"{tag}.depth.f32",
                        "landmarks_mm": (lm * 1e3).round(4).tolist(),
                        "girths_mm": {k: {"z_mm": round(d["z_m"] * 1e3, 4), "contour": round(d["contour_mm"], 4),
                                          "tape": round(d["tape_mm"], 4)} for k, d in g.items()}})
    expected = {
        "data": "SYNTHETIC",
        "scanner": {"views": SCANNER.n_views, "height": SCANNER.height, "width": SCANNER.width,
                    "pixel_m": SCANNER.pixel, "distance_m": SCANNER.distance},
        "landmark_names": list(landmarks.NAMES),
        "depth_scale_m": 0.25,
        "tolerances": TOLERANCES,
        "onnx_vs_torch_max_abs": max(onnx_diff),
        "scans": entries,
    }
    (FIXTURES / "expected.json").write_text(json.dumps(expected, indent=1) + "\n")
    print(f"wrote {FIXTURES}; ONNX vs PyTorch max abs output difference {max(onnx_diff):.2e}")


if __name__ == "__main__":
    main()
