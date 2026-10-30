"""Horizon-safe lag and rolling feature generation.

Guarantees ZERO target leakage by strictly shifting the time series by at least
HORIZON (28 days) before computing any rolling statistics or lags.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    horizon: int = 28,
    lag_days: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """Adds horizon-safe lag and rolling features to the dataframe.
    
    Args:
        df: Input dataframe with ['id', 'date', 'units_sold'].
        horizon: Forecast horizon (default 28). Lags must be >= horizon.
        lag_days: List of lag days (e.g., [28, 35, 42, 364]).
        rolling_windows: Windows for rolling stats on shifted series.
        
    Returns:
        DataFrame with added lag and rolling features.
    """
    if lag_days is None:
        lag_days = [28, 35, 42, 364]
    if rolling_windows is None:
        rolling_windows = [7, 14, 28, 56]

    df = df.sort_values(["id", "date"]).copy()
    g = df.groupby("id", observed=False)["units_sold"]

    # 1. Horizon-safe Lag Features
    for lag in lag_days:
        if lag < horizon:
            raise ValueError(
                f"Data leakage risk: lag {lag} is smaller than forecasting horizon {horizon}!"
            )
        df[f"lag_{lag}"] = g.shift(lag).astype(np.float32)

    # 2. Rolling statistics computed strictly on the series already shifted by HORIZON
    shifted = g.shift(horizon)
    is_zero_shifted = (shifted == 0).astype(np.float32)

    # Fast rolling computation using transform
    for w in rolling_windows:
        roll_mean = shifted.groupby(df["id"], observed=False).transform(
            lambda s: s.rolling(w, min_periods=max(1, w // 2)).mean()
        )
        roll_std = shifted.groupby(df["id"], observed=False).transform(
            lambda s: s.rolling(w, min_periods=max(1, w // 2)).std()
        )
        roll_zero = is_zero_shifted.groupby(df["id"], observed=False).transform(
            lambda s: s.rolling(w, min_periods=max(1, w // 2)).mean()
        )

        df[f"roll_mean_{w}"] = roll_mean.astype(np.float32)
        df[f"roll_std_{w}"] = roll_std.astype(np.float32)
        df[f"roll_zero_share_{w}"] = roll_zero.astype(np.float32)

    return df

