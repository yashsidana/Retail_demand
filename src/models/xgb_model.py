"""Global XGBoost forecaster with Tweedie regression, Optuna tuning, and MLflow logging.

Trained globally across all item-store series simultaneously to borrow cross-series strength
while utilizing categorical embeddings and horizon-safe exogenous variables.
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Optional, Tuple, Any

try:
    import mlflow
    import mlflow.xgboost
except ImportError:
    mlflow = None


class XGBoostForecaster:
    """Global XGBoost model for demand forecasting."""

    def __init__(self, config: Dict):
        self.config = config
        self.model_cfg = config["models"]["xgboost"]
        self.feature_cols: List[str] = []
        self.model: Optional[xgb.Booster] = None
        self.best_iteration: int = 0

    def _prepare_dmatrix(
        self,
        df: pd.DataFrame,
        is_train: bool = True
    ) -> Tuple[xgb.DMatrix, List[str]]:
        """Prepares DMatrix with categorical support and target separation."""
        exclude_cols = ["id", "date", "d", "units_sold", "wm_yr_wk"]
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols].copy()

        # Handle categoricals
        cat_cols = X.select_dtypes(include=["category", "object"]).columns
        for c in cat_cols:
            X[c] = X[c].astype("category")

        y = df["units_sold"].values if "units_sold" in df.columns else None

        dmat = xgb.DMatrix(X, label=y, enable_categorical=True)
        return dmat, feature_cols

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        custom_params: Optional[Dict] = None
    ) -> "XGBoostForecaster":
        """Fits global XGBoost using histogram tree method and early stopping."""
        dtrain, feat_cols = self._prepare_dmatrix(train_df, is_train=True)
        self.feature_cols = feat_cols

        params = {
            "objective": self.model_cfg.get("objective", "reg:tweedie"),
            "tree_method": self.model_cfg.get("tree_method", "hist"),
            "eta": self.model_cfg.get("eta", 0.05),
            "max_depth": self.model_cfg.get("max_depth", 6),
            "min_child_weight": self.model_cfg.get("min_child_weight", 50),
            "subsample": self.model_cfg.get("subsample", 0.8),
            "colsample_bytree": self.model_cfg.get("colsample_bytree", 0.8),
            "lambda": self.model_cfg.get("reg_lambda", 1.0),
            "alpha": self.model_cfg.get("reg_alpha", 0.1),
            "seed": self.config["project"].get("seed", 42),
        }

        if params["objective"] == "reg:tweedie":
            params["tweedie_variance_power"] = self.model_cfg.get("tweedie_variance_power", 1.2)

        if custom_params:
            params.update(custom_params)

        evals = [(dtrain, "train")]
        early_stopping = self.model_cfg.get("early_stopping_rounds", 50)
        n_rounds = self.model_cfg.get("n_estimators", 1000)

        if val_df is not None:
            dval, _ = self._prepare_dmatrix(val_df, is_train=False)
            evals.append((dval, "val"))

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            evals=evals,
            early_stopping_rounds=early_stopping if val_df is not None else None,
            verbose_eval=False
        )

        self.best_iteration = self.model.best_iteration if hasattr(self.model, "best_iteration") else n_rounds
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Generates 28-day forecasts for all rows."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        dtest, _ = self._prepare_dmatrix(test_df, is_train=False)
        preds = self.model.predict(dtest)
        # Demand cannot be negative
        return np.maximum(0.0, preds)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Extracts feature importance table sorted descending."""
        if self.model is None:
            return pd.DataFrame()

        score_dict = self.model.get_score(importance_type=importance_type)
        df_imp = pd.DataFrame([
            {"feature": k, "importance": v} for k, v in score_dict.items()
        ]).sort_values("importance", ascending=False).reset_index(drop=True)

        return df_imp

    def save(self, filepath: str) -> None:
        """Saves model to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_model(filepath)

    def load(self, filepath: str) -> None:
        """Loads model from file."""
        self.model = xgb.Booster()
        self.model.load_model(filepath)

