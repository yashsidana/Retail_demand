"""Generates a realistic M5-compliant sample dataset for local development and testing.

Produces:
- data/raw/calendar.csv (dates, events, SNAP flags per state)
- data/raw/sell_prices.csv (store, item, week, price with promotional drops)
- data/raw/sales_train_validation.csv (wide format d_1 .. d_N with intermittent counts)
"""

import os
import numpy as np
import pandas as pd


def generate_m5_sample(
    output_dir: str = "data/raw",
    n_days: int = 500, # d_1 to d_500
    n_items: int = 25,
    stores: list = None,
    seed: int = 42
):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    if stores is None:
        stores = [
            ("CA_1", "CA"),
            ("CA_2", "CA"),
            ("TX_1", "TX"),
            ("WI_1", "WI")
        ]

    print(f"Generating realistic M5-like dataset ({n_days} days, {n_items} items, {len(stores)} stores)...")

    # 1. Calendar
    start_date = pd.to_datetime("2014-01-01")
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    
    cal_data = []
    events_pool = [
        ("SuperBowl", "Sporting"),
        ("ValentineDay", "Cultural"),
        ("Easter", "Cultural"),
        ("MemorialDay", "National"),
        ("IndependenceDay", "National"),
        ("LaborDay", "National"),
        ("Thanksgiving", "National"),
        ("Christmas", "National"),
        ("NewYear", "National")
    ]

    for idx, dt in enumerate(dates):
        d_str = f"d_{idx + 1}"
        wm_yr_wk = int(f"{dt.year % 100}{dt.isocalendar().week:02d}")
        
        # Random event on ~5% of days
        event1, event_type1 = (None, None)
        if np.random.rand() < 0.05:
            ev = events_pool[np.random.randint(len(events_pool))]
            event1, event_type1 = ev

        # Specific Christmas day shutdown
        if dt.month == 12 and dt.day == 25:
            event1, event_type1 = ("Christmas", "National")

        # SNAP benefit days: roughly days 1-10 of month
        snap_CA = 1 if dt.day <= 10 and np.random.rand() < 0.8 else 0
        snap_TX = 1 if dt.day in [1, 3, 5, 6, 7, 9, 11, 13] else 0
        snap_WI = 1 if dt.day in [2, 3, 5, 8, 11, 12, 14] else 0

        cal_data.append({
            "date": dt.strftime("%Y-%m-%d"),
            "wm_yr_wk": wm_yr_wk,
            "weekday": dt.day_name(),
            "wday": dt.weekday() + 1,
            "month": dt.month,
            "year": dt.year,
            "d": d_str,
            "event_name_1": event1,
            "event_type_1": event_type1,
            "event_name_2": None,
            "event_type_2": None,
            "snap_CA": snap_CA,
            "snap_TX": snap_TX,
            "snap_WI": snap_WI
        })

    df_calendar = pd.DataFrame(cal_data)
    df_calendar.to_csv(os.path.join(output_dir, "calendar.csv"), index=False)
    print(f"-> Saved calendar.csv ({len(df_calendar)} days)")

    # 2. Items metadata
    departments = [
        ("FOODS_1", "FOODS"),
        ("FOODS_2", "FOODS"),
        ("HOBBIES_1", "HOBBIES"),
        ("HOUSEHOLD_1", "HOUSEHOLD")
    ]
    items_meta = []
    for i in range(1, n_items + 1):
        dept_id, cat_id = departments[i % len(departments)]
        item_id = f"{dept_id}_{i:03d}"
        base_price = round(np.random.uniform(1.98, 15.98), 2)
        items_meta.append({
            "item_id": item_id,
            "dept_id": dept_id,
            "cat_id": cat_id,
            "base_price": base_price
        })

    # 3. Sell Prices
    unique_wm_weeks = df_calendar["wm_yr_wk"].unique()
    price_records = []
    for store_id, state_id in stores:
        for it in items_meta:
            b_price = it["base_price"]
            for wk in unique_wm_weeks:
                # 10% chance of a promotional price drop
                if np.random.rand() < 0.10:
                    sell_price = round(b_price * 0.85, 2)
                else:
                    sell_price = b_price
                price_records.append({
                    "store_id": store_id,
                    "item_id": it["item_id"],
                    "wm_yr_wk": wk,
                    "sell_price": sell_price
                })

    df_prices = pd.DataFrame(price_records)
    df_prices.to_csv(os.path.join(output_dir, "sell_prices.csv"), index=False)
    print(f"-> Saved sell_prices.csv ({len(df_prices)} price entries)")

    # 4. Sales Train Validation (Wide format)
    sales_rows = []
    for store_id, state_id in stores:
        for it in items_meta:
            series_id = f"{it['item_id']}_{store_id}_validation"
            row = {
                "id": series_id,
                "item_id": it["item_id"],
                "dept_id": it["dept_id"],
                "cat_id": it["cat_id"],
                "store_id": store_id,
                "state_id": state_id
            }

            # Generate synthetic intermittent sales series
            # Intermittency: zero-inflated Poisson process
            is_food = "FOODS" in it["cat_id"]
            base_lambda = np.random.uniform(1.5, 5.0) if is_food else np.random.uniform(0.2, 1.5)
            zero_prob = 0.35 if is_food else 0.70 # Non-food is much more intermittent!

            for d_idx, dt in enumerate(dates):
                d_col = f"d_{d_idx + 1}"
                
                # Christmas store closure: 0 sales
                if dt.month == 12 and dt.day == 25:
                    row[d_col] = 0
                    continue

                # Weekend multiplier
                w_mult = 1.4 if dt.weekday() in [5, 6] else 1.0
                
                # Zero inflation
                if np.random.rand() < zero_prob:
                    row[d_col] = 0
                else:
                    # Poisson demand count
                    count = np.random.poisson(base_lambda * w_mult)
                    row[d_col] = count

            sales_rows.append(row)

    df_sales = pd.DataFrame(sales_rows)
    df_sales.to_csv(os.path.join(output_dir, "sales_train_validation.csv"), index=False)
    print(f"-> Saved sales_train_validation.csv ({len(df_sales)} item-store series)")


if __name__ == "__main__":
    generate_m5_sample()

