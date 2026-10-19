import numpy as np
import pytest

from body3d import evaluate


@pytest.fixture(scope="module")
def graded(mixed):
    """Every landmark of a body in BMI band i is off by 10*i mm along x."""
    gt = np.zeros((len(mixed), 16, 3))
    pred = gt.copy()
    pred[..., 0] = (mixed.bmi_band * 0.010)[:, None]
    return evaluate.summarise(evaluate.errors_mm(pred, gt), mixed)


def test_errors_mm():
    e = evaluate.errors_mm(np.array([[[0.003, 0.004, 0.0]]]), np.zeros((1, 1, 3)))
    np.testing.assert_allclose(e, [[5.0]])


def test_band_means_and_worst_band(graded):
    bmi = graded["strata"]["bmi"]
    assert [round(g["mean_mm"], 6) for g in bmi.values()] == [0, 10, 20, 30, 40]
    w = graded["worst_bmi_band"]
    assert w["band"] == ">=35" and w["gap_mm"] == pytest.approx(30) and w["ratio"] == pytest.approx(4)
    assert graded["worst_band_any_factor"]["band"] == ">=35"


def test_pck_and_constant_ci(graded):
    assert graded["overall"]["pck10"] == pytest.approx(40)  # bands 0 and 1 of 5
    assert graded["overall"]["pck20"] == pytest.approx(60)
    lo, hi = graded["strata"]["bmi"]["25-30"]["ci95_mm"]
    assert lo == pytest.approx(20) and hi == pytest.approx(20)


def test_bootstrap_ci_brackets_mean(mixed):
    err = np.random.default_rng(1).gamma(2.0, 10.0, (len(mixed), 16))
    o = evaluate.summarise(err, mixed)["overall"]
    assert o["ci95_mm"][0] < o["mean_mm"] < o["ci95_mm"][1]


def test_band_table_renders(graded):
    table = evaluate.band_table([{"run": "x", **graded}])
    assert table.count("\n") == 2 and "4.00x (>=35)" in table
