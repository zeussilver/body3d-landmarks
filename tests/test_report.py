import json

import pytest

from body3d import bodies, report


def _fake(band_means):
    return {"overall": {"mean_mm": sum(band_means) / 5, "pck10": 90.0, "pck20": 99.0},
            "strata": {"bmi": {b: {"mean_mm": m} for b, m in zip(bodies.BMI_LABELS, band_means)}}}


def test_gap_closure_is_paired_by_seed(tmp_path):
    runs = {"e0_s0": [1, 1, 1, 1, 1], "e1_s0": [2, 1, 2, 4, 9], "e2_s0": [2, 1.5, 2, 2, 5],
            "e0_s1": [1, 1, 1, 1, 1], "e1_s1": [2, 1, 2, 4, 11], "e2_s1": [2, 1.5, 2, 2, 6]}
    for name, m in runs.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(_fake(m)))
    s = report.summarise(report.collect(tmp_path))
    assert s["e1_worst_band"] == ">=35"
    g = s["e2_vs_e1"]
    assert g["gap_closure_pct"]["per_seed"] == pytest.approx([50.0, 50.0])
    assert g["in_range_change_mm"]["mean"] == pytest.approx(0.5) and g["A4_met"]
    assert s["conditions"]["e1"]["bmi_mm"][">=35"] == {"mean": 10.0, "std": 1.0}
    assert "E2 closes 50%" in report.table(s)
