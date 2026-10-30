"""Data ingestion and memory optimization module for M5 dataset.

Handles downcasting, wide-to-long unpivoting, and database ingestion.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List


def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Downcasts numeric types and converts string columns to categories to reduce RAM."""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and not pd.api.types.is_categorical_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif str(col_type)[:5] == "float":
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        elif pd.api.types.is_object_dtype(col_type):
            # Check cardinality before converting to category
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    return df


def melt_sales_data(
    sales_df: pd.DataFrame,
    start_day: Optional[int] = None,
    end_day: Optional[int] = None
) -> pd.DataFrame:
    """Unpivots wide M5 sales data (d_1 ... d_N) into long format (d, units_sold).
    
    Args:
        sales_df: Raw wide sales dataframe.
        start_day: Optional first day number (e.g., 1000).
        end_day: Optional last day number (e.g., 1913).
        
    Returns:
        pd.DataFrame in long format.
    """
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales_df.columns if c.startswith("d_")]
    
    if start_day is not None or end_day is not None:
        filtered_days = []
        for c in day_cols:
            d_num = int(c.split("_")[1])
            if (start_day is None or d_num >= start_day) and (end_day is None or d_num <= end_day):
                filtered_days.append(c)
        day_cols = filtered_days

    melted = pd.melt(
        sales_df,
        id_vars=id_vars,
        value_vars=day_cols,
        var_name="d",
        value_name="units_sold"
    )
    melted["units_sold"] = melted["units_sold"].astype(np.int16)
    return melted


def load_raw_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads raw CSV files from data directory with optional subsetting.
    
    Returns:
        Tuple of (sales_df, calendar_df, prices_df)
    """
    raw_dir = config["data"]["raw_dir"]
    sales_path = os.path.join(raw_dir, config["data"]["sales_file"])
    calendar_path = os.path.join(raw_dir, config["data"]["calendar_file"])
    prices_path = os.path.join(raw_dir, config["data"]["prices_file"])

    if not (os.path.exists(sales_path) and os.path.exists(calendar_path) and os.path.exists(prices_path)):
        raise FileNotFoundError(
            f"Raw CSV files missing in {raw_dir}. Run `python scripts/generate_sample_data.py` "
            f"or download the official M5 dataset from Kaggle."
        )

    print(f"Loading calendar from {calendar_path}...")
    calendar_df = pd.read_csv(calendar_path)
    calendar_df = downcast_dataframe(calendar_df)

    print(f"Loading prices from {prices_path}...")
    prices_df = pd.read_csv(prices_path)
    prices_df = downcast_dataframe(prices_df)

    print(f"Loading sales from {sales_path}...")
    sales_df = pd.read_csv(sales_path)

    # Optional subsetting for fast development
    sample_stores = config["data"].get("sample_stores")
    if sample_stores:
        sales_df = sales_df[sales_df["store_id"].isin(sample_stores)].copy()
        prices_df = prices_df[prices_df["store_id"].isin(sample_stores)].copy()

    sample_items = config["data"].get("sample_items_count")
    if sample_items and sample_items < sales_df["item_id"].nunique():
        unique_items = sales_df["item_id"].unique()[:sample_items]
        sales_df = sales_df[sales_df["item_id"].isin(unique_items)].copy()
        prices_df = prices_df[prices_df["item_id"].isin(unique_items)].copy()

    return sales_df, calendar_df, prices_df


def save_to_sqlite(
    sales_long: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    db_path: str
) -> None:
    """Saves raw ingested tables to SQLite database for SQL processing."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    
    # Read schema
    schema_path = os.path.join(os.path.dirname(__file__), "sql", "schema.sql")
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            conn.executescript(f.read())
            
    sales_long.to_sql("sales_long", conn, if_exists="replace", index=False)
    calendar.to_sql("calendar", conn, if_exists="replace", index=False)
    prices.to_sql("prices", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"Successfully saved tables to SQLite database at {db_path}")

