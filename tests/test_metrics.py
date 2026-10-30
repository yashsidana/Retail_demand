"""Unit tests for intermittent demand forecasting metrics."""

import pytest
import numpy as np
from src.evaluation.metrics import calc_mae, calc_rmsse, calc_wape, calc_bias, calc_pinball_loss


def test_mae_standard():
    y_true = np.array([10.0, 0.0, 5.0, 2.0])
    y_pred = np.array([8.0, 1.0, 5.0, 0.0])
    # abs diff: [2, 1, 0, 2] -> mean = 1.25
    assert np.isclose(calc_mae(y_true, y_pred), 1.25)


def test_wape_stability_on_zeros():
    """WAPE must be stable and defined even when actual sales contain many zeros."""
    y_true = np.array([0.0, 0.0, 0.0, 4.0])
    y_pred = np.array([1.0, 0.0, 0.0, 3.0])
    # sum(|diff|) = 1 + 0 + 0 + 1 = 2; sum(y_true) = 4; WAPE = 2/4 = 0.50
    assert np.isclose(calc_wape(y_true, y_pred), 0.50)

    # When all actuals are zero, it should return mean absolute error instead of crashing
    y_all_zero = np.zeros(5)
    y_pred_zero = np.ones(5)
    wape_val = calc_wape(y_all_zero, y_pred_zero)
    assert not np.isnan(wape_val)
    assert not np.isinf(wape_val)


def test_bias_direction():
    """Signed bias should be positive for over-forecasting and negative for under-forecasting."""
    y_true = np.array([5.0, 5.0, 5.0])
    y_over = np.array([8.0, 7.0, 6.0])
    assert calc_bias(y_true, y_over) > 0.0 # over-forecasts

    y_under = np.array([2.0, 3.0, 4.0])
    assert calc_bias(y_true, y_under) < 0.0 # under-forecasts


def test_rmsse_hand_computed():
    """Test RMSSE against a hand-calculated scenario.
    
    Training history: [2, 4, 6] -> diffs: [2, 2] -> denominator MSE = 4.0
    Evaluation: actual = [8, 10], pred = [7, 9] -> errors: [1, 1] -> numerator MSE = 1.0
    RMSSE = sqrt(1.0 / 4.0) = 0.50
    """
    y_train = np.array([2.0, 4.0, 6.0])
    y_true = np.array([8.0, 10.0])
    y_pred = np.array([7.0, 9.0])

    rmsse = calc_rmsse(y_true, y_pred, y_train)
    assert np.isclose(rmsse, 0.50)


def test_rmsse_constant_history():
    """When training history is constant (denominator diffs = 0), RMSSE should not divide by zero."""
    y_train = np.array([0.0, 0.0, 0.0, 0.0])
    y_true = np.array([1.0, 1.0])
    y_pred = np.array([0.0, 0.0])

    rmsse = calc_rmsse(y_true, y_pred, y_train)
    assert not np.isnan(rmsse)
    assert not np.isinf(rmsse)


def test_pinball_loss():
    """Tests quantile loss at 0.90 quantile."""
    y_true = np.array([10.0])
    y_pred_under = np.array([8.0]) # under-prediction: loss = 0.90 * 2 = 1.8
    assert np.isclose(calc_pinball_loss(y_true, y_pred_under, quantile=0.90), 1.8)

    y_pred_over = np.array([12.0]) # over-prediction: loss = (0.90 - 1) * (-2) = 0.20
    assert np.isclose(calc_pinball_loss(y_true, y_pred_over, quantile=0.90), 0.2)

