"""ARIMA and SARIMAX forecaster implementation.

Includes Augmented Dickey-Fuller (ADF) & KPSS stationarity tests, seasonal differencing,
and AIC-based model selection.
Applied to aggregated store/dept levels and representative item-store samples.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def check_stationarity(series: pd.Series, significance: float = 0.05) -> Dict[str, Union[bool, float]]:
    """Performs both ADF and KPSS unit root tests.
    
    - ADF H0: Non-stationary (unit root). Reject if p-value < 0.05.
    - KPSS H0: Stationary. Reject if p-value < 0.05.
    """
    clean_series = series.dropna()
    if len(clean_series) < 20:
        return {"stationary": True, "adf_pvalue": 0.0, "kpss_pvalue": 0.1}

    # ADF test
    adf_res = adfuller(clean_series, autolag="AIC")
    adf_p = float(adf_res[1])
    is_adf_stationary = adf_p < significance

    # KPSS test
    try:
        kpss_res = kpss(clean_series, regression="c", nlags="auto")
        kpss_p = float(kpss_res[1])
        is_kpss_stationary = kpss_p >= significance
    except Exception:
        kpss_p = 0.05
        is_kpss_stationary = True

    # Combined decision: stationary if ADF rejects unit root and KPSS does not reject stationarity
    stationary = is_adf_stationary and is_kpss_stationary

    return {
        "stationary": stationary,
        "adf_pvalue": adf_p,
        "kpss_pvalue": kpss_p
    }


class ARIMAForecaster:
    """Per-series ARIMA/SARIMAX forecaster with automatic differencing and AIC selection."""

    def __init__(
        self,
        seasonal_period: int = 7,
        max_p: int = 2,
        max_d: int = 1,
        max_q: int = 2
    ):
        self.seasonal_period = seasonal_period
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.fitted_models: Dict[str, any] = {}

    def fit_single_series(self, series: pd.Series) -> Optional[any]:
        """Finds best SARIMAX order by minimizing AIC."""
        clean = series.dropna().astype(float)
        if len(clean) < 28 or clean.sum() == 0:
            return None

        # Check stationarity
        stat_info = check_stationarity(clean)
        d = 0 if stat_info["stationary"] else 1

        best_aic = float("inf")
        best_model = None

        # Grid search over small orders for efficiency
        for p in range(0, self.max_p + 1):
            for q in range(0, self.max_q + 1):
                try:
                    mod = SARIMAX(
                        clean,
                        order=(p, d, q),
                        seasonal_order=(1, 0, 0, self.seasonal_period),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    res = mod.fit(disp=False, maxiter=50)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_model = res
                except Exception:
                    continue

        return best_model

    def fit(
        self,
        df_train: pd.DataFrame,
        id_col: str = "id",
        target_col: str = "units_sold",
        max_series: Optional[int] = 10
    ):
        """Fits SARIMAX models on unique series (or sampled subset)."""
        unique_ids = df_train[id_col].unique()
        if max_series and len(unique_ids) > max_series:
            unique_ids = unique_ids[:max_series]

        for sid in unique_ids:
            s_data = df_train[df_train[id_col] == sid].sort_values("date").set_index("date")[target_col]
            model_res = self.fit_single_series(s_data)
            if model_res is not None:
                self.fitted_models[sid] = model_res

        return self

    def predict(
        self,
        df_test: pd.DataFrame,
        id_col: str = "id",
        horizon: int = 28
    ) -> np.ndarray:
        """Forecasts horizon steps ahead per series."""
        preds = []
        for sid, group in df_test.groupby(id_col, sort=False):
            n_steps = len(group)
            if sid in self.fitted_models:
                try:
                    forecast = self.fitted_models[sid].forecast(steps=n_steps)
                    forecast_vals = np.maximum(0.0, np.array(forecast))
                except Exception:
                    forecast_vals = np.zeros(n_steps, dtype=np.float32)
            else:
                forecast_vals = np.zeros(n_steps, dtype=np.float32)
            preds.extend(forecast_vals.tolist())

        return np.array(preds, dtype=np.float32)

