"""Heuristic baseline models for retail demand forecasting.

Provides benchmark comparisons: Naive, Seasonal Naive (7-day cycle), and Moving Average.
Any complex machine learning model must convincingly beat these baselines to justify deployment.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union


class NaiveForecaster:
    """Predicts future days equal to the last observed value in history."""

    def __init__(self):
        self.last_values: Dict[str, float] = {}

    def fit(self, df_train: pd.DataFrame, id_col: str = "id", target_col: str = "units_sold"):
        last_rows = df_train.sort_values("date").groupby(id_col)[target_col].last()
        self.last_values = last_rows.to_dict()
        return self

    def predict(self, df_test: pd.DataFrame, id_col: str = "id") -> np.ndarray:
        preds = df_test[id_col].map(self.last_values).fillna(0.0).values
        return np.maximum(0.0, preds)


class SeasonalNaiveForecaster:
    """Predicts each future day matching the same day of the week from the previous cycle (7 days)."""

    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        self.seasonal_profiles: Dict[str, Dict[int, float]] = {}

    def fit(self, df_train: pd.DataFrame, id_col: str = "id", target_col: str = "units_sold"):
        df = df_train.copy()
        df["wday"] = pd.to_datetime(df["date"]).dt.weekday
        # Average recent sales for each weekday per series
        recent_cutoff = df["date"].max()
        df_recent = df[pd.to_datetime(df["date"]) >= (pd.to_datetime(recent_cutoff) - pd.Timedelta(days=28))]
        
        profiles = df_recent.groupby([id_col, "wday"], observed=False)[target_col].mean().to_dict()
        self.seasonal_profiles = profiles
        # Fallback overall mean per series
        self.fallback = df.groupby(id_col, observed=False)[target_col].mean().to_dict()
        return self

    def predict(self, df_test: pd.DataFrame, id_col: str = "id") -> np.ndarray:
        df = df_test.copy()
        df["wday"] = pd.to_datetime(df["date"]).dt.weekday
        
        preds = []
        for _, row in df.iterrows():
            key = (row[id_col], row["wday"])
            val = self.seasonal_profiles.get(key, self.fallback.get(row[id_col], 0.0))
            preds.append(val)
            
        return np.maximum(0.0, np.array(preds, dtype=np.float32))


class MovingAverageForecaster:
    """Predicts future demand as the rolling mean of the recent trailing window (e.g. 28 days)."""

    def __init__(self, window: int = 28):
        self.window = window
        self.window_means: Dict[str, float] = {}

    def fit(self, df_train: pd.DataFrame, id_col: str = "id", target_col: str = "units_sold"):
        df = df_train.sort_values("date").copy()
        last_date = pd.to_datetime(df["date"].max())
        cutoff = last_date - pd.Timedelta(days=self.window)
        
        df_window = df[pd.to_datetime(df["date"]) > cutoff]
        means = df_window.groupby(id_col, observed=False)[target_col].mean().to_dict()
        self.window_means = means
        return self

    def predict(self, df_test: pd.DataFrame, id_col: str = "id") -> np.ndarray:
        preds = df_test[id_col].map(self.window_means).fillna(0.0).values
        return np.maximum(0.0, preds)

