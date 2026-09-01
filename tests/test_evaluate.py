import pytest

from compagnon_immo.models.evaluate import evaluate_regression


def test_evaluate_regression_returns_expected_metrics():
    y_true = [1000.0, 2000.0, 3000.0]
    y_pred = [1000.0, 2000.0, 3000.0]

    metrics = evaluate_regression(y_true, y_pred)

    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)