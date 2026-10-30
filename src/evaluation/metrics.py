"""Evaluation metrics for intermittent retail demand forecasting.

Includes MAE, RMSSE (M5 scaled metric), WRMSSE, WAPE, signed Bias, and Pinball loss.
Notice: MAPE is intentionally omitted because it is undefined when actual sales are 0!
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional


def calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (in unit sales)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def calc_rmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: np.ndarray
) -> float:
    """Root Mean Squared Scaled Error (RMSSE).
    
    Divides out-of-sample MSE by the in-sample one-step naive benchmark MSE.
    
    Formula:
        RMSSE = sqrt( (1/h * sum((y_t - y_hat_t)^2)) / (1/(n-1) * sum((y_t - y_{t-1})^2)) )
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_train_history = np.asarray(y_train_history, dtype=np.float64)

    h = len(y_true)
    if h == 0:
        return 0.0

    numerator = np.mean((y_true - y_pred) ** 2)

    # In-sample naive difference squared
    if len(y_train_history) > 1:
        diffs = np.diff(y_train_history)
        denominator = np.mean(diffs ** 2)
    else:
        denominator = 1.0

    if denominator <= 1e-8:
        # If train series had constant sales, scale by 1.0 to prevent division by zero
        denominator = 1.0

    return float(np.sqrt(numerator / denominator))


def calc_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (WAPE).
    
    WAPE = sum(|y_true - y_pred|) / sum(y_true)
    Unlike MAPE, WAPE does not divide individual observations by zero.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    total_actual = np.sum(y_true)
    if total_actual == 0:
        return float(np.mean(np.abs(y_pred)))
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual)


def calc_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Signed mean forecast error (Forecast - Actual).
    
    Positive bias = over-forecasting (excess holding cost / waste).
    Negative bias = under-forecasting (stockouts / lost sales).
    """
    return float(np.mean(y_pred - y_true))


def calc_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.95) -> float:
    """Quantile / Pinball loss for safety stock and uncertainty evaluation.
    
    L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true - y_pred
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def calc_wrmsse(
    df_eval: pd.DataFrame,
    id_col: str = "id",
    actual_col: str = "actual",
    pred_col: str = "predicted",
    revenue_col: Optional[str] = "revenue"
) -> float:
    """Calculates revenue-weighted or equal-weighted RMSSE across multiple series."""
    grouped = df_eval.groupby(id_col)
    rmsse_list = []
    weight_list = []

    for name, group in grouped:
        y_t = group[actual_col].values
        y_p = group[pred_col].values
        # Use first portion or actuals as proxy history if not supplied separately
        history = y_t[:max(1, len(y_t)//2)]
        score = calc_rmsse(y_t, y_p, history)
        rmsse_list.append(score)
        
        if revenue_col and revenue_col in group.columns:
            w = group[revenue_col].sum()
        else:
            w = 1.0
        weight_list.append(w)

    weights = np.array(weight_list)
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(len(rmsse_list)) / len(rmsse_list)

    return float(np.sum(np.array(rmsse_list) * weights))


def evaluate_forecasts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Returns a consolidated dictionary of all standard forecasting metrics."""
    if y_train_history is None:
        y_train_history = y_true

    return {
        "MAE": calc_mae(y_true, y_pred),
        "RMSSE": calc_rmsse(y_true, y_pred, y_train_history),
        "WAPE": calc_wape(y_true, y_pred),
        "Bias": calc_bias(y_true, y_pred),
        "Pinball_95": calc_pinball_loss(y_true, y_pred, quantile=0.95)
    }

