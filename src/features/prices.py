"""Price and promotion feature engineering.

Extracts relative pricing, week-over-week price changes, and promotional discount flags.
"""

import pandas as pd
import numpy as np


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds pricing and promotional features.
    
    - sell_price: raw unit price
    - price_mean: historical mean price per item-store
    - relative_price: sell_price / price_mean (measures relative discount)
    - price_change_wow: week-over-week price difference
    - is_promo: binary flag if price is >= 10% lower than historical average
    """
    df = df.copy()
    
    if "sell_price" not in df.columns:
        return df

    # Historical mean price per item-store
    price_mean = df.groupby(["store_id", "item_id"], observed=False)["sell_price"].transform("mean")
    df["price_mean"] = price_mean.astype(np.float32)

    # Relative price (normalizes items with different base prices)
    df["relative_price"] = (df["sell_price"] / (df["price_mean"] + 1e-5)).astype(np.float32)

    # Week-over-week price change: sorted by item, store, wm_yr_wk
    # Note: prices change weekly
    df = df.sort_values(["id", "date"])
    df["price_lag_1w"] = df.groupby("id", observed=False)["sell_price"].shift(7)
    df["price_change_wow"] = (df["sell_price"] - df["price_lag_1w"]).fillna(0.0).astype(np.float32)
    df.drop(columns=["price_lag_1w"], inplace=True)

    # Promotion flag: discount > 10% from historical mean
    df["is_promo"] = (df["relative_price"] < 0.90).astype(np.int8)

    # Handle missing prices (e.g., item not active yet)
    df["is_price_missing"] = df["sell_price"].isna().astype(np.int8)
    df["sell_price"] = df["sell_price"].fillna(0.0).astype(np.float32)
    df["relative_price"] = df["relative_price"].fillna(1.0).astype(np.float32)

    return df

