"""Leakage validation test suite.

Ensures no information from the forecast horizon leaks into features or training folds.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.lags import add_lag_and_rolling_features
from src.evaluation.rolling_origin import RollingOriginCV


def test_horizon_safe_lag_shift():
    """Asserts that all lag and rolling features are shifted by at least the 28-day horizon."""
    horizon = 28
    dates = pd.date_range("2016-01-01", periods=100, freq="D")
    
    # Create simple series with predictable incremental sales: 1, 2, 3, ...
    df = pd.DataFrame({
        "id": ["item_1"] * 100,
        "date": dates.astype(str),
        "units_sold": np.arange(1, 101, dtype=np.float32)
    })

    df_feats = add_lag_and_rolling_features(df, horizon=horizon, lag_days=[28, 35], rolling_windows=[7])

    # Check row 50 (date 2016-02-20, index 49, sales = 50)
    # lag_28 must equal sales on day (50 - 28) = 22
    assert df_feats.loc[49, "lag_28"] == 22.0

    # roll_mean_7 on row 50 must be computed on days shifted by 28:
    # shifted values at row 49 correspond to sales at row 49 - 28 = index 21 (sales 22)
    # window of 7 ends at day 22: [16, 17, 18, 19, 20, 21, 22] -> mean = 19.0
    assert np.isclose(df_feats.loc[49, "roll_mean_7"], 19.0)


def test_strict_temporal_split_no_overlap():
    """Verifies that rolling origin folds have zero overlap between train and validation dates."""
    dates = pd.date_range("2016-01-01", periods=150, freq="D")
    df = pd.DataFrame({
        "date": dates.astype(str),
        "val": np.random.randn(150)
    })

    cv = RollingOriginCV(horizon_days=28, n_folds=3, date_column="date")
    
    for train_df, val_df, fold_name in cv.split(df):
        train_max_date = pd.to_datetime(train_df["date"]).max()
        val_min_date = pd.to_datetime(val_df["date"]).min()
        val_max_date = pd.to_datetime(val_df["date"]).max()

        # Training data must strictly precede validation data
        assert train_max_date < val_min_date, f"Leakage detected in {fold_name}!"
        # Validation window length must equal horizon
        val_days = len(pd.to_datetime(val_df["date"]).unique())
        assert val_days == 28, f"Expected 28 days in validation window, got {val_days}"


def test_no_leakage_exception_on_invalid_lag():
    """Attempting to use a lag smaller than the horizon must raise ValueError."""
    df = pd.DataFrame({
        "id": ["item_1"] * 50,
        "date": pd.date_range("2016-01-01", periods=50).astype(str),
        "units_sold": np.ones(50)
    })

    with pytest.raises(ValueError, match="Data leakage risk"):
        # Setting lag 14 with horizon 28
        add_lag_and_rolling_features(df, horizon=28, lag_days=[14, 28])

