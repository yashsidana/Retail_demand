"""Master feature generation pipeline.

Integrates horizon-safe lags, calendar, events, SNAP flags, and pricing dynamics.
Saves the engineered tabular dataset in high-performance Apache Parquet format.
"""

import os
import pandas as pd
from typing import Dict, Optional
from .lags import add_lag_and_rolling_features
from .calendar import add_calendar_features
from .prices import add_price_features
from ..data.load import downcast_dataframe


def build_all_features(
    df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """Builds all feature sets according to configuration."""
    feat_cfg = config["features"]
    horizon = feat_cfg.get("horizon", 28)
    lag_days = feat_cfg.get("lag_days", [28, 35, 42, 364])
    rolling_windows = feat_cfg.get("rolling_windows", [7, 14, 28, 56])

    print("Building horizon-safe lag and rolling features...")
    df = add_lag_and_rolling_features(
        df,
        horizon=horizon,
        lag_days=lag_days,
        rolling_windows=rolling_windows
    )

    if feat_cfg.get("include_calendar_features", True):
        print("Building calendar and event proximity features...")
        df = add_calendar_features(df)

    if feat_cfg.get("include_price_features", True):
        print("Building price and promotion features...")
        df = add_price_features(df)

    # Encode categorical columns
    cat_cols = feat_cfg.get("categorical_columns", ["item_id", "dept_id", "cat_id", "store_id", "state_id"])
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Drop rows that have NaN in the primary horizon lag (e.g., first 28 days)
    primary_lag = f"lag_{horizon}"
    if primary_lag in df.columns:
        initial_len = len(df)
        df = df.dropna(subset=[primary_lag]).reset_index(drop=True)
        print(f"Dropped {initial_len - len(df)} initial warmup rows with missing horizon lags.")

    df = downcast_dataframe(df)
    return df


def run_feature_pipeline(config: Dict, fact_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Runs the feature pipeline and persists Parquet file."""
    if fact_df is None:
        parquet_in = os.path.join(config["data"]["processed_dir"], "fact_sales.parquet")
        if not os.path.exists(parquet_in):
            raise FileNotFoundError(f"Fact table {parquet_in} not found. Run build_fact_table first.")
        print(f"Reading fact table from {parquet_in}...")
        fact_df = pd.read_parquet(parquet_in)

    features_df = build_all_features(fact_df, config)

    out_file = config["data"]["parquet_feature_file"]
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    features_df.to_parquet(out_file, index=False)
    print(f"Successfully saved engineered features to {out_file} (Shape: {features_df.shape})")
    
    return features_df

