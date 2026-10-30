"""Calendar and event feature engineering.

Generates temporal features: day-of-week, month, week-of-year, weekend flag,
event categories, and holiday distance proximity features.
"""

import pandas as pd
import numpy as np


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts date components and event proximity features."""
    df = df.copy()
    dt = pd.to_datetime(df["date"])

    # Temporal components
    df["wday"] = dt.dt.weekday.astype(np.int8) # 0=Monday, 6=Sunday
    df["mday"] = dt.dt.day.astype(np.int8)
    df["week"] = dt.dt.isocalendar().week.astype(np.int8)
    df["month"] = dt.dt.month.astype(np.int8)
    df["year"] = dt.dt.year.astype(np.int16)
    df["is_weekend"] = df["wday"].isin([5, 6]).astype(np.int8)

    # Event labels cleaning
    for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype("category")

    # Holiday proximity: compute days to next and days since last holiday
    # A day is a holiday if event_name_1 is not 'None'
    unique_dates = df[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    if "event_name_1" in df.columns:
        date_events = df[["date", "event_name_1"]].drop_duplicates("date")
        holiday_dates = pd.to_datetime(
            date_events[date_events["event_name_1"] != "None"]["date"]
        ).sort_values().values
    else:
        holiday_dates = np.array([], dtype="datetime64[ns]")

    if len(holiday_dates) > 0:
        dt_vals = pd.to_datetime(df["date"]).values
        
        # Days since last holiday
        # For each date, search sorted in holiday_dates
        idx_prev = np.searchsorted(holiday_dates, dt_vals, side="right") - 1
        valid_prev = idx_prev >= 0
        days_since = np.full(len(df), 999.0, dtype=np.float32)
        days_since[valid_prev] = (
            (dt_vals[valid_prev] - holiday_dates[idx_prev[valid_prev]]).astype("timedelta64[D]").astype(float)
        )
        
        # Days to next holiday
        idx_next = np.searchsorted(holiday_dates, dt_vals, side="left")
        valid_next = idx_next < len(holiday_dates)
        days_to = np.full(len(df), 999.0, dtype=np.float32)
        days_to[valid_next] = (
            (holiday_dates[idx_next[valid_next]] - dt_vals[valid_next]).astype("timedelta64[D]").astype(float)
        )
        
        df["days_since_last_event"] = days_since
        df["days_to_next_event"] = days_to
    else:
        df["days_since_last_event"] = 999.0
        df["days_to_next_event"] = 999.0

    return df

