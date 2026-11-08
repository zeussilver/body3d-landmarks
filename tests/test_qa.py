import json
from types import SimpleNamespace

import numpy as np
import pytest

from body3d import bodies, qa, qa_rules, qa_vlm
from body3d.scanner import BACKGROUND


@pytest.fixture(scope="module")
def arm_w():
    return bodies.arm_weights()


def _scan(template, faces, lm_idx, arm_w, label, seed=0):
    v = template.vertices[0].astype(np.float64)
    return qa.qa_scan(v, faces, v[lm_idx], label, np.random.default_rng(seed), arm_w)


def test_assign_is_balanced_and_rebuilds_only_arms_down():
    b, labels = qa.assign(2, seed=9)
    assert np.bincount(labels).tolist() == [2] * len(qa.CLASSES)
    down = labels == qa.CLASSES.index("arms_down")
    assert np.all((b.pose[down, :2] >= qa.ARMS_DOWN_DEG[0]) & (b.pose[down, :2] <= qa.ARMS_DOWN_DEG[1]))
    assert np.all(np.abs(b.pose[~down, :2]) <= bodies.ARM_JITTER_DEG)


def test_ok_scan_passes_every_rule(template, faces, lm_idx, arm_w):
    depth, _, info = _scan(template, faces, lm_idx, arm_w, "ok")
    f = qa_rules.features(depth)
    assert info == {}
    assert f["border_px"] < 20 and f["drop_asym"] < 0.1 and f["move_sym"] < 0.12 and f["arm_gap"] > 0.75


def test_cropped_scan_really_leaves_the_image(template, faces, lm_idx, arm_w):
    depth, _, info = _scan(template, faces, lm_idx, arm_w, "cropped")
    assert info["offset_m"] >= qa.CROP_OFFSET_M[0]
    assert qa_rules.features(depth)["border_px"] > 20


def test_moved_scan_breaks_the_mirror_symmetry(template, faces, lm_idx, arm_w):
    depth, _, info = _scan(template, faces, lm_idx, arm_w, "moved", seed=2)
    assert qa.MOVE_VIEWS[0] <= len(info["views"]) <= qa.MOVE_VIEWS[1]
    assert qa_rules.features(depth)["move_sym"] > 0.12


def test_dropout_scan_records_what_was_removed(template, faces, lm_idx, arm_w):
    depth, _, info = _scan(template, faces, lm_idx, arm_w, "dropout", seed=4)
    assert 1 <= len(info["dropped"]) <= 2
    assert {d["kind"] for d in info["dropped"]} <= set(qa.DROP_KINDS)


def test_two_opposite_blank_views_are_still_a_dropout(template, faces, lm_idx, arm_w):
    depth, _, _ = _scan(template, faces, lm_idx, arm_w, "ok")
    depth[2] = depth[6] = BACKGROUND  # a pair: the mirror comparison alone sees nothing
    assert qa_rules.features(depth)["drop_asym"] == 1.0


def test_arms_down_closes_the_arm_gap(template, faces, lm_idx, arm_w):
    pose = template.pose.copy()
    pose[:, :2] = 45.0
    down = bodies.build(template.phenotype, template.local, pose)
    depth, _, _ = _scan(down, faces, lm_idx, arm_w, "ok")
    assert qa_rules.features(depth)["arm_gap"] < 0.5


def test_metrics_count_invalid_answers_as_wrong_and_flagged():
    y = np.array([0, 0, 1, 2, 3, 4])
    pred = np.array([0, -1, 1, 0, 3, 4])
    m = qa.metrics(y, pred, n_boot=10)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["invalid_answers"] == 1
    assert m["rescan"]["false_alarms_on_ok"] == pytest.approx(0.5)  # the invalid answer on an ok scan
    assert m["rescan"]["failures_caught"] == pytest.approx(3 / 4)  # the moved scan called ok is missed
    assert m["confusion"]["matrix"][0][-1] == 1


def test_rule_cascade_takes_the_first_rule_that_fires():
    x = np.array([[100, 0, 0.9, 0.9, 0.0],   # border fires first -> cropped
                  [0, 0, 0.9, 0.9, 0.0],     # drop_asym -> dropout
                  [0, 0, 0.0, 0.9, 0.0],     # move_sym -> moved
                  [0, 0, 0.0, 0.0, 0.1],     # arm_gap low -> arms_down
                  [0, 0, 0.0, 0.0, 0.9]])    # nothing -> ok
    th = {0: 20, 1: 600, 2: 0.15, 3: 0.13, 4: 0.75}
    assert [qa.CLASSES[c] for c in qa_rules.classify(x, th)] == ["cropped", "dropout", "moved", "arms_down", "ok"]


def test_vlm_parse():
    assert qa_vlm.parse(json.dumps({"label": "moved", "reason": "45 deg looks frontal"})) == (2, "45 deg looks frontal")
    assert qa_vlm.parse(json.dumps({"label": "blurry", "reason": "x"}))[0] == -1
    assert qa_vlm.parse("not json")[0] == -1


class _FakeMessages:
    def __init__(self, texts):
        self.texts, self.calls = list(texts), []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.texts.pop(0))], stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=500, output_tokens=30), model=kw["model"],
                               _request_id="req_test")


def test_vlm_classify_resends_once_on_an_invalid_label(tmp_path):
    pytest.importorskip("anthropic")  # in the optional "qa" dependency group
    img = tmp_path / "scan.png"
    qa.grid_image(np.full((8, 256, 160), BACKGROUND, np.float32)).save(img)
    fake = _FakeMessages([json.dumps({"label": "blurry", "reason": "?"}),
                          json.dumps({"label": "dropout", "reason": "all views blank"})])
    rec = qa_vlm.classify(SimpleNamespace(messages=fake), "claude-sonnet-5-5", "low", img)
    assert rec["pred"] == qa.CLASSES.index("dropout") and rec["attempts"] == 2
    assert rec["input_tokens"] == 1000 and rec["model_served"] == "claude-sonnet-5-5"
    sent = fake.calls[0]
    assert sent["output_config"]["format"]["schema"]["properties"]["label"]["enum"] == list(qa.CLASSES)
    assert sent["messages"][0]["content"][0]["type"] == "image"


@pytest.fixture
def fake_run(tmp_path, monkeypatch):
    """qa_vlm.run against a fake split of 6 scans and a stubbed classify()."""
    anthropic = pytest.importorskip("anthropic")  # in the optional "qa" dependency group

    monkeypatch.setattr(qa_vlm, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(qa, "load_split", lambda split: {"labels": np.zeros(6, np.int64)})
    monkeypatch.setattr(qa, "write_images", lambda split: None)
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: SimpleNamespace(base_url="https://api.anthropic.com"))
    return tmp_path


def test_vlm_run_stops_when_the_first_calls_all_fail(fake_run, monkeypatch):
    calls = []
    monkeypatch.setattr(qa_vlm, "classify", lambda *a: calls.append(1) or {"pred": -1, "error": "AuthenticationError 401",
                                                                           "attempts": 1})
    with pytest.raises(SystemExit, match="first 3 calls failed"):
        qa_vlm.run("claude-sonnet-5-5", "low", "test", 1, 2, None)
    assert len(calls) == 3


def test_vlm_resume_retries_errors_and_keeps_the_latest_answer(fake_run, monkeypatch):
    answers = iter([{"pred": 0, "attempts": 1}, {"pred": -1, "error": "APIConnectionError", "attempts": 1}]
                   + [{"pred": 0, "attempts": 1}] * 4 + [{"pred": 0, "attempts": 1}])
    monkeypatch.setattr(qa_vlm, "classify", lambda *a: next(answers))
    qa_vlm.run("claude-sonnet-5-5", "low", "test", 1, 1, None)  # scan 1 hits an error
    qa_vlm.run("claude-sonnet-5-5", "low", "test", 1, 1, None)  # only scan 1 is called again
    rows = qa_vlm.load_runs()[("claude-sonnet-5-5", "low", "test")][0]
    assert [r["i"] for r in rows] == list(range(6))
    assert all("error" not in r for r in rows)


def test_few_shot_examples_come_from_dev_and_end_with_a_cache_breakpoint(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(qa, "load_split", lambda split: seen.append(split) or {"labels": np.array([4, 3, 2, 1, 0, 0])})
    monkeypatch.setattr(qa, "write_images", lambda split: None)
    img = tmp_path / "x.png"
    qa.grid_image(np.full((8, 256, 160), BACKGROUND, np.float32)).save(img)
    monkeypatch.setattr(qa, "image_path", lambda split, i: img)
    blocks = qa_vlm.examples(1)
    assert seen == ["dev"]
    assert len(blocks) == 1 + 2 * len(qa.CLASSES)  # intro, then image + label per class
    assert [b["text"].split(".")[0] for b in blocks[2::2]] == [f"Label: {c}" for c in qa.CLASSES]
    assert blocks[-1]["cache_control"] == {"type": "ephemeral"} and qa_vlm.examples(0) == []


def test_endpoint_kind_never_records_the_host():
    assert qa_vlm.endpoint_kind(SimpleNamespace(base_url="https://api.anthropic.com")) == "anthropic"
    assert qa_vlm.endpoint_kind(SimpleNamespace(base_url="https://example-relay.invalid/v1")) == "relay"
