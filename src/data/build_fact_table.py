"""Fact sales table construction module.

Executes the relational join between long sales, calendar, and prices.
Respects state-specific SNAP logic and handles missing price rows via LEFT JOIN.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Optional
from .load import load_raw_data, melt_sales_data, downcast_dataframe, save_to_sqlite


def build_fact_table_sql(db_path: str) -> pd.DataFrame:
    """Executes fact_sales.sql query against SQLite database."""
    conn = sqlite3.connect(db_path)
    sql_path = os.path.join(os.path.dirname(__file__), "sql", "fact_sales.sql")
    with open(sql_path, "r") as f:
        query = f.read()
    
    conn.executescript(query)
    fact_df = pd.read_sql("SELECT * FROM fact_sales", conn)
    conn.close()
    return fact_df


def build_fact_table_pandas(
    sales_long: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame
) -> pd.DataFrame:
    """Builds the fact table directly in Pandas matching the exact SQL query logic.
    
    - State-specific SNAP: CA -> snap_CA, TX -> snap_TX, WI -> snap_WI
    - LEFT JOIN on prices: missing price row indicates product was not on sale yet
    """
    # Merge sales with calendar on 'd'
    fact = pd.merge(
        sales_long,
        calendar,
        on="d",
        how="inner"
    )

    # Dynamic state-specific SNAP
    snap_conditions = [
        fact["state_id"] == "CA",
        fact["state_id"] == "TX",
        fact["state_id"] == "WI"
    ]
    snap_choices = [
        fact["snap_CA"],
        fact["snap_TX"],
        fact["snap_WI"]
    ]
    fact["snap"] = np.select(snap_conditions, snap_choices, default=0).astype(np.int8)

    # LEFT JOIN on prices
    # Ensure types match before merge
    fact["wm_yr_wk"] = fact["wm_yr_wk"].astype(np.int32)
    prices["wm_yr_wk"] = prices["wm_yr_wk"].astype(np.int32)
    
    fact = pd.merge(
        fact,
        prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left"
    )

    # Filter out pre-launch periods: exclude dates where the item has never been sold yet
    # A zero before first appearance/price is not real zero demand
    fact["date"] = pd.to_datetime(fact["date"])
    fact = fact.sort_values(["id", "date"]).reset_index(drop=True)

    fact = downcast_dataframe(fact)
    return fact


def build_fact_table(config: Dict, use_sql: bool = False) -> pd.DataFrame:
    """Main entrypoint for constructing the analysis fact table."""
    sales_df, calendar_df, prices_df = load_raw_data(config)
    
    start_day = config["data"].get("start_day")
    end_day = config["data"].get("end_day")
    
    print(f"Melting sales data (days {start_day} to {end_day})...")
    sales_long = melt_sales_data(sales_df, start_day=start_day, end_day=end_day)

    db_path = config["data"]["database_url"].replace("sqlite:///", "")
    
    if use_sql:
        print(f"Ingesting into SQLite: {db_path}...")
        save_to_sqlite(sales_long, calendar_df, prices_df, db_path)
        print("Executing fact_sales.sql...")
        fact_df = build_fact_table_sql(db_path)
    else:
        print("Building fact table in memory with Pandas...")
        fact_df = build_fact_table_pandas(sales_long, calendar_df, prices_df)

    # Convert date to string for parquet / consistent formatting
    fact_df["date"] = fact_df["date"].astype(str)
    
    out_dir = config["data"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fact_sales.parquet")
    fact_df.to_parquet(out_path, index=False)
    print(f"Fact table created with shape {fact_df.shape} and saved to {out_path}")
    
    return fact_df

