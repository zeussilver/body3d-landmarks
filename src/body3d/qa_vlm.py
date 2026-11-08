"""Claude as a zero-shot vision-language model on the scan check.

Each scan image (the same 2 x 4 grid the other methods see) goes to Claude with a
description of the scanner and the five classes; the answer is constrained to a JSON
schema. Every call is recorded (label, reason, latency, tokens, model served), so scores
can be recomputed without calling the API again, and an interrupted run resumes.

    uv sync --group qa
    python -m body3d.qa_vlm run --model claude-sonnet-5-5 --repeats 3       # calls the API
    python -m body3d.qa_vlm score                                             # no API calls

Credentials come from the environment, as for any Anthropic SDK client.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from body3d import qa
from body3d.data import RESULTS

RUNS_DIR = RESULTS / "qa_vlm"
PRICES = {  # USD per million input / output tokens, list prices (2026-10)
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
MAX_TOKENS = 8000  # room for adaptive thinking; the answer itself is ~50 tokens
SUBSET = 100  # models too slow or costly for the whole test split run on its first 100 scans (20/20/20/19/21)
NO_EFFORT = {"claude-haiku-4-5", "claude-haiku-4-5-20251001"}  # models that reject output_config.effort; recorded as effort "none"
SYSTEM = """You check 3D body scans for problems before the person leaves the scanner, so a bad scan can be redone at once.

How the scan is made: the person stands on a turntable in front of a fixed depth camera, holding the scan pose: upright, feet apart, arms held away from the body in an A-shape. The camera captures 8 orthographic depth images, one every 45 degrees of rotation. The image you get shows them in a 2 x 4 grid: the top row is 0, 45, 90 and 135 degrees, the bottom row is 180, 225, 270 and 315 degrees, so each bottom image shows the person from exactly the opposite side of the image above it. Brighter means closer to the camera; black is background. Small black specks and dots on the body are normal sensor noise.

If the person stood still and centred, each bottom image is a left-right mirror image of the outline above it, and the person turns evenly by 45 degrees from one image to the next.

Classify the scan as exactly one of:
- ok: none of the problems below.
- arms_down: the arms hang down against the sides of the body instead of being held away from it.
- moved: the person turned or stepped during the scan, so in 2 or 3 neighbouring images they face a different direction or stand in a different place than the even rotation predicts (for example a 45-degree image that looks frontal, or a bottom image that is not the mirror of the one above it).
- cropped: part of the body is cut off by the left or right edge of an image because the person stood off-centre.
- dropout: in one or two images a large part of the body is missing: the image is blank, or a wide band or a large patch of the body is black, much bigger than the normal small specks.

If more than one seems to apply, pick the most obvious one. Give a short reason (under 25 words) naming the image or images you based it on."""
# One line per example; each only restates the class definition above (nothing here was
# written from test-set mistakes).
EXAMPLE_NOTES = {
    "ok": "None of the problems.",
    "arms_down": "The arms hang down against the sides of the body.",
    "moved": "The person turned or stepped during some neighbouring views.",
    "cropped": "Part of the body is cut off by the image edge in some views.",
    "dropout": "A large part of the body is missing in one or two views.",
}
SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": list(qa.CLASSES)}, "reason": {"type": "string"}},
    "required": ["label", "reason"],
    "additionalProperties": False,
}


def run_path(model: str, effort: str, split: str, repeat: int) -> Path:
    return RUNS_DIR / f"{model}__{effort}__{split}__r{repeat}.jsonl"


def endpoint_kind(client) -> str:
    """'anthropic' or 'relay'; the host itself is never recorded."""
    host = str(client.base_url).split("//")[-1].split("/")[0]
    return "anthropic" if host == "api.anthropic.com" or host.endswith(".anthropic.com") else "relay"


def parse(text: str) -> tuple[int, str]:
    """Model text -> (class index or -1, reason)."""
    try:
        d = json.loads(text)
        return qa.CLASSES.index(d["label"]), str(d.get("reason", ""))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return -1, text[:200]


def _image_block(path: Path) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def examples(shots: int) -> list[dict]:
    """``shots`` labelled dev scans per class (the first of each class), as message blocks.

    The last block carries a cache breakpoint: the examples are the same for every scan,
    so after the first call they are read from the prompt cache.
    """
    if shots == 0:
        return []
    s = qa.load_split("dev")
    qa.write_images("dev")
    blocks = [{"type": "text", "text": f"Labelled examples ({shots} per class) from earlier scans:"}]
    for c, name in enumerate(qa.CLASSES):
        for i in np.flatnonzero(s["labels"] == c)[:shots]:
            blocks += [_image_block(qa.image_path("dev", int(i))),
                       {"type": "text", "text": f"Label: {name}. {EXAMPLE_NOTES[name]}"}]
    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


def classify(client, model: str, effort: str, image: Path, shots: list[dict] | None = None) -> dict:
    """One scan, with one fresh resend if the answer is not a valid label. Never raises for API errors."""
    import anthropic

    lead = (shots or []) + ([{"type": "text", "text": "Now the scan to check:"}] if shots else [])
    content = lead + [_image_block(image), {"type": "text", "text": "Classify this scan."}]
    rec = {"attempts": 0, "latency_s": 0.0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
           "cache_write_tokens": 0, "max_tokens": MAX_TOKENS}
    output_config = {"format": {"type": "json_schema", "schema": SCHEMA}}
    if model not in NO_EFFORT:
        output_config["effort"] = effort
    for _ in range(2):
        rec["attempts"] += 1
        t0 = time.perf_counter()
        try:
            r = client.messages.create(
                model=model, max_tokens=MAX_TOKENS, system=SYSTEM,
                messages=[{"role": "user", "content": content}],
                output_config=output_config,
            )
        except anthropic.APIStatusError as e:  # 4xx/5xx after the SDK's own retries
            rec.update(latency_s=rec["latency_s"] + time.perf_counter() - t0, pred=-1,
                       error=f"{type(e).__name__} {e.status_code}")
            continue
        except anthropic.APIConnectionError as e:
            rec.update(latency_s=rec["latency_s"] + time.perf_counter() - t0, pred=-1, error=type(e).__name__)
            continue
        rec["latency_s"] += time.perf_counter() - t0
        rec["input_tokens"] += r.usage.input_tokens
        rec["output_tokens"] += r.usage.output_tokens
        rec["cache_read_tokens"] += getattr(r.usage, "cache_read_input_tokens", None) or 0
        rec["cache_write_tokens"] += getattr(r.usage, "cache_creation_input_tokens", None) or 0
        text = next((b.text for b in r.content if b.type == "text"), "")
        pred, reason = parse(text) if r.stop_reason == "end_turn" else (-1, f"stop_reason={r.stop_reason}")
        rec.update(pred=pred, reason=reason, stop_reason=r.stop_reason, model_served=r.model,
                   request_id=r._request_id)
        rec.pop("error", None)
        if pred >= 0:
            break
    return rec


def run(model: str, effort: str, split: str, repeats: int, workers: int, limit: int | None, shots: int = 0) -> None:
    import anthropic

    if model in NO_EFFORT:
        effort = "none"
    shot_blocks = examples(shots)
    tag = effort if shots == 0 else f"{effort}+{shots}shot"
    client = anthropic.Anthropic(max_retries=4)
    s = qa.load_split(split)
    qa.write_images(split)
    n = len(s["labels"]) if limit is None else min(limit, len(s["labels"]))
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    kind = endpoint_kind(client)
    print(f"endpoint: {kind}", flush=True)
    for rep in range(repeats):
        path = run_path(model, tag, split, rep)
        last = {r["i"]: r for r in _rows(path)}
        done = {i for i, r in last.items() if "error" not in r}  # rows that hit an API error are retried
        todo = [i for i in range(n) if i not in done]
        print(f"{path.name}: {len(done)} recorded, {len(todo)} to go", flush=True)

        def one(i: int) -> dict:
            rec = classify(client, model, effort, qa.image_path(split, i), shot_blocks)
            return {"i": i, "true": int(s["labels"][i]), **rec, "endpoint": kind, "effort": effort, "shots": shots}

        with path.open("a") as f:
            # The first few scans go one at a time: if they all fail (bad key, unknown model),
            # stop instead of recording a whole run of errors. (This also writes the prompt
            # cache before the parallel calls read it.)
            head, rest = todo[:3], todo[3:]
            first = [one(i) for i in head]
            for rec in first:
                f.write(json.dumps(rec) + "\n")
            f.flush()
            if first and all("error" in r for r in first):
                raise SystemExit(f"stopping: the first {len(first)} calls failed ({first[0]['error']})")
            with ThreadPoolExecutor(workers) as ex:
                for k, rec in enumerate(ex.map(one, rest), start=len(head)):
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    if (k + 1) % 25 == 0:
                        print(f"  {k + 1}/{len(todo)}", flush=True)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def load_runs() -> dict[tuple[str, str, str], list[list[dict]]]:
    """{(model, effort, split): [rows of repeat 0, rows of repeat 1, ...]}, rows sorted by scan index."""
    out: dict = {}
    for p in sorted(RUNS_DIR.glob("*.jsonl")):
        model, effort, split, _ = p.stem.split("__")
        last = {r["i"]: r for r in _rows(p)}  # a retried scan keeps its latest answer
        out.setdefault((model, effort, split), []).append([last[i] for i in sorted(last)])
    return out


def _vote(preds: np.ndarray) -> np.ndarray:
    """Majority label over repeats (rows = repeats); ties go to the lowest class index."""
    return np.array([np.bincount(col[col >= 0], minlength=len(qa.CLASSES)).argmax() if (col >= 0).any() else -1
                     for col in preds.T])


def score(subset: int = SUBSET) -> dict:
    """Metrics per (model, effort, split).

    - ``full``: the first repeat that covers every scan of the split (None if none does);
    - ``first_<subset>``: the first ``subset`` scans of repeat 0, for models run on fewer scans;
    - ``repeats``: agreement on the scans every repeat covers.
    Latency, tokens and cost use every recorded call.
    """
    res = {}
    for (model, effort, split), reps in load_runs().items():
        labels = qa.load_split(split)["labels"]
        n_split = len(labels)
        by_i = [{r["i"]: r for r in rows} for rows in reps]
        rows_all = [r for rows in reps for r in rows]
        complete = [d for d in by_i if len(d) == n_split]
        out = {"model": model, "effort": effort, "split": split,
               "scans_per_repeat": [len(d) for d in by_i],
               "endpoint": sorted({r["endpoint"] for r in rows_all}),
               "models_served": sorted({r.get("model_served", "?") for r in rows_all})}
        if complete:
            out["full"] = qa.metrics(labels, np.array([complete[0][i]["pred"] for i in range(n_split)]))
        head = by_i[0]
        if all(i in head for i in range(subset)):
            out[f"first_{subset}"] = qa.metrics(labels[:subset], np.array([head[i]["pred"] for i in range(subset)]))
        common = sorted(set.intersection(*(set(d) for d in by_i))) if len(by_i) > 1 else []
        if common:
            preds = np.array([[d[i]["pred"] for i in common] for d in by_i])
            y = labels[common]
            out["repeats"] = {"n_repeats": len(by_i), "scans": len(common),
                              "same_label_all_repeats": float((preds == preds[0]).all(0).mean()),
                              "accuracy_per_repeat": [float((p == y).mean()) for p in preds],
                              "majority_vote_accuracy": float((_vote(preds) == y).mean())}
        lat = np.array([r["latency_s"] for r in rows_all])
        tin = float(np.mean([r["input_tokens"] for r in rows_all]))
        tout = float(np.mean([r["output_tokens"] for r in rows_all]))
        tcr = float(np.mean([r.get("cache_read_tokens", 0) for r in rows_all]))
        tcw = float(np.mean([r.get("cache_write_tokens", 0) for r in rows_all]))
        pin, pout = PRICES.get(model, (np.nan, np.nan))
        out.update({
            "calls": len(rows_all),
            "latency_s": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))},
            "tokens_per_scan": {"input": tin, "output": tout, "cache_read": tcr, "cache_write": tcw},
            # Every prompt token at the full input price, no cache discount: through the relay the
            # cache counts covered the per-scan image too, which Anthropic's prefix cache cannot do,
            # so they are not trusted. An upper bound for the few-shot runs; zero-shot has no cache.
            "usd_per_1000_scans_list_price": float(((tin + tcr + tcw) * pin + tout * pout) / 1e3),
            "resends": int(sum(r["attempts"] - 1 for r in rows_all)),
            "api_errors": int(sum("error" in r for r in rows_all)),
        })
        res[f"{model}/{effort}/{split}"] = out
        main_m = out.get("full") or out.get(f"first_{subset}")
        tag = "full" if "full" in out else f"first {subset}"
        if main_m:
            print(f"{model}/{effort}/{split} [{tag}] {qa.summary_line(main_m)}")
        if "repeats" in out:
            rp = out["repeats"]
            print(f"   repeats {rp['n_repeats']} on {rp['scans']} scans: same label {rp['same_label_all_repeats']:.2f}, "
                  f"acc {rp['accuracy_per_repeat']}")
        print(f"   p50 {out['latency_s']['p50']:.1f}s p95 {out['latency_s']['p95']:.1f}s, "
              f"prompt/output tokens {tin + tcr + tcw:.0f}/{tout:.0f}, ${out['usd_per_1000_scans_list_price']:.2f} per 1,000 scans")
    (RESULTS / "scan_qa_vlm.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="call the API and record answers")
    r.add_argument("--model", default="claude-sonnet-5-5")
    r.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"],
                   help="ignored for models in NO_EFFORT")
    r.add_argument("--split", default="test", choices=list(qa.SPLITS))
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--limit", type=int, default=None, help="first N scans only (a cheap probe)")
    r.add_argument("--shots", type=int, default=0, help="labelled dev examples per class in the prompt")
    sub.add_parser("score", help="score recorded runs, no API calls")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.model, a.effort, a.split, a.repeats, a.workers, a.limit, a.shots)
    else:
        score()


if __name__ == "__main__":
    main()
