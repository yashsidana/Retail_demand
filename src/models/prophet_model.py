"""Facebook Prophet forecaster wrapper.

Additive decomposition: y(t) = Trend + Weekly Seasonality + Annual Seasonality + Events/SNAP.
Adds exogenous regressors (SNAP, promotional discounts) and clips negative predictions.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Optional, List

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
except ImportError:
    Prophet = None


class ProphetForecaster:
    """Wrapper around Facebook Prophet with exogenous regressors for retail series."""

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_mode: str = "multiplicative",
        include_regressors: bool = True
    ):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.include_regressors = include_regressors
        self.models: Dict[str, any] = {}

    def fit_single_series(self, df_series: pd.DataFrame) -> Optional[any]:
        if Prophet is None:
            return None

        # Prophet expects 'ds' and 'y'
        p_df = pd.DataFrame({
            "ds": pd.to_datetime(df_series["date"]),
            "y": df_series["units_sold"].astype(float)
        })

        if self.include_regressors:
            if "snap" in df_series.columns:
                p_df["snap"] = df_series["snap"].fillna(0).astype(float)
            if "relative_price" in df_series.columns:
                p_df["relative_price"] = df_series["relative_price"].fillna(1.0).astype(float)

        m = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode,
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False
        )

        if self.include_regressors:
            if "snap" in p_df.columns:
                m.add_regressor("snap")
            if "relative_price" in p_df.columns:
                m.add_regressor("relative_price")

        try:
            m.fit(p_df)
            return m
        except Exception:
            return None

    def fit(
        self,
        df_train: pd.DataFrame,
        id_col: str = "id",
        max_series: Optional[int] = 10
    ):
        """Fits Prophet on unique series (or sampled subset)."""
        unique_ids = df_train[id_col].unique()
        if max_series and len(unique_ids) > max_series:
            unique_ids = unique_ids[:max_series]

        for sid in unique_ids:
            s_data = df_train[df_train[id_col] == sid]
            m = self.fit_single_series(s_data)
            if m is not None:
                self.models[sid] = m

        return self

    def predict(
        self,
        df_test: pd.DataFrame,
        id_col: str = "id"
    ) -> np.ndarray:
        """Generates future predictions and clips at zero."""
        preds = []
        for sid, group in df_test.groupby(id_col, sort=False):
            n_steps = len(group)
            if sid in self.models:
                p_future = pd.DataFrame({
                    "ds": pd.to_datetime(group["date"])
                })
                if self.include_regressors:
                    if "snap" in group.columns:
                        p_future["snap"] = group["snap"].fillna(0).astype(float).values
                    if "relative_price" in group.columns:
                        p_future["relative_price"] = group["relative_price"].fillna(1.0).astype(float).values
                try:
                    forecast = self.models[sid].predict(p_future)
                    vals = np.maximum(0.0, forecast["yhat"].values)
                except Exception:
                    vals = np.zeros(n_steps, dtype=np.float32)
            else:
                vals = np.zeros(n_steps, dtype=np.float32)
            preds.extend(vals.tolist())

        return np.array(preds, dtype=np.float32)

