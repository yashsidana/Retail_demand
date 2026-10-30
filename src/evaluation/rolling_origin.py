"""Rolling-origin (expanding window) time-based cross-validation.

Strictly preserves chronological ordering to prevent forward-looking data leakage.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Generator


class RollingOriginCV:
    """Rolling-origin expanding window time series cross-validator."""

    def __init__(
        self,
        horizon_days: int = 28,
        n_folds: int = 3,
        date_column: str = "date"
    ):
        self.horizon_days = horizon_days
        self.n_folds = n_folds
        self.date_column = date_column

    def split(
        self,
        df: pd.DataFrame
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, str], None, None]:
        """Yields (train_df, val_df, cutoff_date) for each rolling fold.
        
        The training window expands chronologically.
        Validation windows are strictly non-overlapping, contiguous 28-day blocks.
        """
        unique_dates = np.sort(pd.to_datetime(df[self.date_column]).unique())
        total_days = len(unique_dates)
        
        required_days = (self.n_folds + 1) * self.horizon_days
        if total_days < required_days:
            raise ValueError(
                f"Insufficient days ({total_days}) for {self.n_folds} folds of horizon {self.horizon_days}."
            )

        # Cutoff dates moving forward in time
        # Last available date is T
        last_idx = total_days - 1
        
        for fold in range(self.n_folds):
            # Calculate validation window boundaries
            # Fold 0 (earliest fold) to Fold (n_folds - 1) (latest fold)
            offset = (self.n_folds - fold) * self.horizon_days
            val_end_idx = last_idx - offset + self.horizon_days
            val_start_idx = val_end_idx - self.horizon_days + 1
            train_end_idx = val_start_idx - 1

            cutoff_date = unique_dates[train_end_idx]
            val_start_date = unique_dates[val_start_idx]
            val_end_date = unique_dates[val_end_idx]

            df_dates = pd.to_datetime(df[self.date_column])
            train_mask = df_dates <= cutoff_date
            val_mask = (df_dates >= val_start_date) & (df_dates <= val_end_date)

            train_df = df[train_mask].copy()
            val_df = df[val_mask].copy()

            fold_name = f"Fold_{fold+1}_cutoff_{str(cutoff_date)[:10]}"
            yield train_df, val_df, fold_name

    @staticmethod
    def get_final_holdout_split(
        df: pd.DataFrame,
        horizon_days: int = 28,
        date_column: str = "date"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        """Returns the final untouched holdout split (last 28 days as test)."""
        unique_dates = np.sort(pd.to_datetime(df[date_column]).unique())
        cutoff_date = unique_dates[-horizon_days - 1]
        test_start_date = unique_dates[-horizon_days]

        df_dates = pd.to_datetime(df[date_column])
        train_df = df[df_dates <= cutoff_date].copy()
        test_df = df[df_dates >= test_start_date].copy()

        return train_df, test_df, str(cutoff_date)[:10]

