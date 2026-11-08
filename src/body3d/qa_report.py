"""Why the check matters, and one table for all methods.

    python -m body3d.qa_report impact     # E0 landmark error on each class of QA test scan
    python -m body3d.qa_report summary    # results/scan_qa_summary.json + README table
"""

from __future__ import annotations

import json
import sys

import numpy as np

from body3d import evaluate, qa
from body3d.data import RESULTS, ROOT


def impact(run_dir: str = "e0_s0") -> dict:
    """Landmark error of the full-range model (E0, seed 0) on QA test scans, by class.

    A failure is worth catching if it ruins the measurement; this checks which ones do.
    """
    import torch

    from body3d.model import LandmarkModel
    from body3d.train import device, predict

    s = qa.load_split("test")
    model = LandmarkModel().to(device())
    model.load_state_dict(torch.load(ROOT / "runs" / run_dir / "best.pt", map_location=device()))
    pred = predict(model, {"depth_mm": s["depth_mm"], "landmarks": s["landmarks"]})
    err = evaluate.errors_mm(pred, s["landmarks"])  # (N, K)
    per_body = err.mean(1)
    res = {"model": f"E0 ({run_dir}), trained on ok-style scans only", "unit": "mean 3D landmark error, mm",
           "by_class": {}}
    for c, name in enumerate(qa.CLASSES):
        e = per_body[s["labels"] == c]
        res["by_class"][name] = {"n": int(len(e)), "mean_mm": float(e.mean()), "median_mm": float(np.median(e)),
                                 "p90_mm": float(np.percentile(e, 90)), "pct_over_10mm": float((e > 10).mean() * 100)}
    (RESULTS / "scan_qa_landmarks.json").write_text(json.dumps(res, indent=1) + "\n")
    for name, v in res["by_class"].items():
        print(f"{name:10s} mean {v['mean_mm']:6.2f}  median {v['median_mm']:6.2f}  p90 {v['p90_mm']:6.2f}  >10mm {v['pct_over_10mm']:5.1f}%")
    return res


def _fmt(r: dict) -> str:
    acc = f"{r['accuracy']:.3f}" + (f" ± {r['accuracy_sd']:.3f}" if "accuracy_sd" in r else "")
    caught = f"{r['failures_caught']:.2f}" if r.get("failures_caught") is not None else ""
    fa = f"{r['false_alarms_on_ok']:.2f}" if r.get("false_alarms_on_ok") is not None else ""
    same = (f"{r['same_label_all_repeats']:.2f} ({r['repeat_n']} × {r['repeat_scans']} scans)"
            if r.get("same_label_all_repeats") is not None else "")
    return (f"| {r['method']} | {r['labels_used']} | {acc} | {r['macro_f1']:.3f} | {caught} | {fa} | "
            f"{r['time_per_scan']} | {r['usd_per_1000_scans']} | {same} |")


def _row(name: str, labels: str, m: dict, time_per_scan: str, usd: str, **extra) -> dict:
    return {"method": name, "labels_used": labels, "n": m["n"], "accuracy": m["accuracy"],
            "accuracy_ci95": m["accuracy_ci95"], "macro_f1": m["macro_f1"],
            "failures_caught": m["rescan"]["failures_caught"], "false_alarms_on_ok": m["rescan"]["false_alarms_on_ok"],
            "recall": {c: v["recall"] for c, v in m["per_class"].items()}, "time_per_scan": time_per_scan,
            "usd_per_1000_scans": usd, **extra}


def summary() -> dict:
    """Two tables: every method on all 250 test scans, and on the first 100 (where Opus was run)."""
    from body3d.qa_vlm import SUBSET

    test = qa.load_split("test")
    y = test["labels"]
    rules = json.loads((RESULTS / "scan_qa_rules.json").read_text())
    clip = json.loads((RESULTS / "scan_qa_clip.json").read_text())
    vlm = json.loads((RESULTS / "scan_qa_vlm.json").read_text()) if (RESULTS / "scan_qa_vlm.json").exists() else {}
    preds = {"rules": np.load(qa.QA_DIR / "test" / "pred_rules.npy"),
             "probe": np.load(qa.QA_DIR / "test" / "pred_clip_probe.npy"),
             "zero_shot": np.load(qa.QA_DIR / "test" / "pred_clip_zero_shot.npy")}
    clip_ms = f"{clip['embed_ms_per_scan']:.0f} ms ({clip['device'].upper()})"
    rules_ms = f"{rules['ms_per_scan_cpu']:.0f} ms (CPU)"
    full, sub = [], []
    for table, sl in ((full, slice(None)), (sub, slice(0, SUBSET))):
        def m(p):
            return qa.metrics(y[sl], p[sl])
        table.append(_row("Rules (turntable geometry)", "500 dev scans set 5 thresholds", m(preds["rules"]), rules_ms, "0"))
        table.append(_row("CLIP ViT-B/32 zero-shot", "none", m(preds["zero_shot"]), clip_ms, "0"))
        if table is full:
            for k in (10, 20):
                cv = clip["probe_learning_curve_test"][str(k)]
                table.append({"method": "CLIP + linear probe", "labels_used": f"{k} per class ({5 * k})",
                              "accuracy": cv["accuracy_mean"], "accuracy_sd": cv["accuracy_sd"],
                              "macro_f1": cv["macro_f1_mean"], "time_per_scan": clip_ms, "usd_per_1000_scans": "0"})
        table.append(_row("CLIP + linear probe", "100 per class (500)", m(preds["probe"]), clip_ms, "0"))
        for v in vlm.values():
            if v["split"] != "test":
                continue
            mm = v.get("full") if table is full else v.get(f"first_{SUBSET}")
            if mm is None:
                continue
            rp = v.get("repeats", {})
            extra = {"endpoint": v["endpoint"], "models_served": v["models_served"]}
            if table is full and rp:
                extra.update(accuracy_per_repeat=rp["accuracy_per_repeat"], same_label_all_repeats=rp["same_label_all_repeats"],
                             repeat_n=rp["n_repeats"], repeat_scans=rp["scans"],
                             majority_vote_accuracy=rp["majority_vote_accuracy"])
            effort, _, shots = v["effort"].partition("+")
            mode = f"{shots[0]} example per class" if shots else "zero-shot"
            effort_txt = f", effort {effort}" if effort != "none" else ""
            labels = f"{5 * int(shots[0])} dev examples in the prompt" if shots else "none"
            table.append(_row(f"{v['model']} {mode}{effort_txt}", labels, mm,
                              f"{v['latency_s']['p50']:.0f} s p50 / {v['latency_s']['p95']:.0f} s p95 (API)",
                              f"{v['usd_per_1000_scans_list_price']:.2f}", **extra))
    out = {"test_all": full, f"test_first_{SUBSET}": sub}
    (RESULTS / "scan_qa_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    head = ("| method | labels used | accuracy | macro-F1 | failures caught | false alarms on ok | time / scan | "
            "$ / 1,000 scans | same label in every repeat |\n|---|---|---|---|---|---|---|---|---|")
    for name, table in out.items():
        print(f"\n{name}\n{head}")
        for r in table:
            print(_fmt(r))
    return out


if __name__ == "__main__":
    {"impact": impact, "summary": summary}[sys.argv[1] if len(sys.argv) > 1 else "summary"]()
