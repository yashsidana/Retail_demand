"""Master End-to-End Execution Pipeline.

Orchestrates:
1. Ingestion and Fact Table Creation
2. Horizon-Safe Feature Engineering (Parquet)
3. Rolling-Origin Time-Based Cross Validation
4. Benchmarking Ladder: Baselines vs ARIMA vs Prophet vs Global XGBoost
5. MLflow Tracking and Metrics Logging
6. PuLP Inventory Linear Program Optimization
7. Simulation Replay: Business Cost Reduction & Service Level Benchmark
8. Exporting Artifacts for FastAPI Serving & Power BI Analytics
"""

import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.data.build_fact_table import build_fact_table
from src.features.build_features import run_feature_pipeline
from src.evaluation.rolling_origin import RollingOriginCV
from src.evaluation.metrics import evaluate_forecasts, calc_rmsse, calc_mae, calc_bias, calc_wape
from src.models.baselines import NaiveForecaster, SeasonalNaiveForecaster, MovingAverageForecaster
from src.models.arima import ARIMAForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.xgb_model import XGBoostForecaster
from src.optimization.inventory_lp import InventoryLPOptimizer, compute_safety_stock_and_rop
from src.optimization.simulate_policy import simulate_inventory_policy, compare_policies

try:
    import mlflow
except ImportError:
    mlflow = None


def load_config(config_path: str = "config/default.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(config_path: str = "config/default.yaml"):
    config = load_config(config_path)
    print("=" * 70)
    print("STARTING RETAIL DEMAND FORECASTING & INVENTORY OPTIMIZATION PIPELINE")
    print("=" * 70)

    # 1. Ingestion & Fact Table
    print("\n--- STAGE 1: Ingestion & Relational Fact Table Construction ---")
    fact_parquet = os.path.join(config["data"]["processed_dir"], "fact_sales.parquet")
    if not os.path.exists(fact_parquet):
        fact_df = build_fact_table(config, use_sql=False)
    else:
        print(f"Loading existing fact table from {fact_parquet}...")
        fact_df = pd.read_parquet(fact_parquet)

    # 2. Horizon-safe Feature Engineering
    print("\n--- STAGE 2: Horizon-Safe Feature Engineering ---")
    feat_parquet = config["data"]["parquet_feature_file"]
    if not os.path.exists(feat_parquet):
        feat_df = run_feature_pipeline(config, fact_df)
    else:
        print(f"Loading existing features from {feat_parquet}...")
        feat_df = pd.read_parquet(feat_parquet)

    # 3. Rolling-Origin Time-Based Cross Validation
    print("\n--- STAGE 3: Rolling-Origin Time-Based Cross Validation ---")
    cv = RollingOriginCV(
        horizon_days=config["validation"]["horizon_days"],
        n_folds=config["validation"]["n_folds"],
        date_column="date"
    )

    models_scores = {
        "Naive": {"rmsse": [], "mae": [], "bias": []},
        "Seasonal Naive": {"rmsse": [], "mae": [], "bias": []},
        "Moving Average": {"rmsse": [], "mae": [], "bias": []},
        "SARIMA (Sample)": {"rmsse": [], "mae": [], "bias": []},
        "Prophet (Sample)": {"rmsse": [], "mae": [], "bias": []},
        "XGBoost (Tweedie)": {"rmsse": [], "mae": [], "bias": []}
    }

    fold_idx = 0
    all_validation_residuals = []

    for train_df, val_df, fold_name in cv.split(feat_df):
        fold_idx += 1
        print(f"\n>>> Running {fold_name} (Train rows: {len(train_df)}, Val rows: {len(val_df)})")

        y_val = val_df["units_sold"].values

        # 3a. Baselines
        naive = NaiveForecaster().fit(train_df)
        preds_naive = naive.predict(val_df)
        models_scores["Naive"]["rmsse"].append(calc_rmsse(y_val, preds_naive, train_df["units_sold"].values))
        models_scores["Naive"]["mae"].append(calc_mae(y_val, preds_naive))
        models_scores["Naive"]["bias"].append(calc_bias(y_val, preds_naive))

        s_naive = SeasonalNaiveForecaster().fit(train_df)
        preds_s_naive = s_naive.predict(val_df)
        models_scores["Seasonal Naive"]["rmsse"].append(calc_rmsse(y_val, preds_s_naive, train_df["units_sold"].values))
        models_scores["Seasonal Naive"]["mae"].append(calc_mae(y_val, preds_s_naive))
        models_scores["Seasonal Naive"]["bias"].append(calc_bias(y_val, preds_s_naive))

        ma = MovingAverageForecaster(window=28).fit(train_df)
        preds_ma = ma.predict(val_df)
        models_scores["Moving Average"]["rmsse"].append(calc_rmsse(y_val, preds_ma, train_df["units_sold"].values))
        models_scores["Moving Average"]["mae"].append(calc_mae(y_val, preds_ma))
        models_scores["Moving Average"]["bias"].append(calc_bias(y_val, preds_ma))

        # 3b. Local Models (Sampled series due to scaling constraints)
        sample_ids = train_df["id"].unique()[:5]
        train_sample = train_df[train_df["id"].isin(sample_ids)]
        val_sample = val_df[val_df["id"].isin(sample_ids)]
        y_val_sample = val_sample["units_sold"].values

        arima = ARIMAForecaster(max_p=1, max_q=1).fit(train_sample, max_series=5)
        preds_arima = arima.predict(val_sample)
        models_scores["SARIMA (Sample)"]["rmsse"].append(calc_rmsse(y_val_sample, preds_arima, train_sample["units_sold"].values))
        models_scores["SARIMA (Sample)"]["mae"].append(calc_mae(y_val_sample, preds_arima))
        models_scores["SARIMA (Sample)"]["bias"].append(calc_bias(y_val_sample, preds_arima))

        prophet = ProphetForecaster().fit(train_sample, max_series=5)
        preds_prophet = prophet.predict(val_sample)
        models_scores["Prophet (Sample)"]["rmsse"].append(calc_rmsse(y_val_sample, preds_prophet, train_sample["units_sold"].values))
        models_scores["Prophet (Sample)"]["mae"].append(calc_mae(y_val_sample, preds_prophet))
        models_scores["Prophet (Sample)"]["bias"].append(calc_bias(y_val_sample, preds_prophet))

        # 3c. Global XGBoost (Tweedie loss)
        xgb_mod = XGBoostForecaster(config).fit(train_df, val_df)
        preds_xgb = xgb_mod.predict(val_df)
        models_scores["XGBoost (Tweedie)"]["rmsse"].append(calc_rmsse(y_val, preds_xgb, train_df["units_sold"].values))
        models_scores["XGBoost (Tweedie)"]["mae"].append(calc_mae(y_val, preds_xgb))
        models_scores["XGBoost (Tweedie)"]["bias"].append(calc_bias(y_val, preds_xgb))

        # Store residuals for safety stock estimation
        residuals = y_val - preds_xgb
        all_validation_residuals.extend(residuals.tolist())

    # Build Model Comparison Table
    print("\n" + "=" * 75)
    print("MODEL COMPARISON TABLE (Mean across Rolling-Origin Folds)")
    print("=" * 75)
    table_rows = []
    for model_name, metrics in models_scores.items():
        table_rows.append({
            "Model": model_name,
            "RMSSE (Mean)": round(float(np.mean(metrics["rmsse"])), 4),
            "MAE (Units)": round(float(np.mean(metrics["mae"])), 4),
            "Bias (Mean Error)": round(float(np.mean(metrics["bias"])), 4),
            "Fold Spread (RMSSE Std)": round(float(np.std(metrics["rmsse"])), 4)
        })
    df_results = pd.DataFrame(table_rows)
    print(df_results.to_string(index=False))

    # 4. Final Train & 28-Day Forecast Generation
    print("\n--- STAGE 4: Final Model Training & 28-Day Production Forecast ---")
    train_holdout, test_holdout, cutoff_str = RollingOriginCV.get_final_holdout_split(
        feat_df,
        horizon_days=config["validation"]["horizon_days"]
    )
    print(f"Training final XGBoost model up to cutoff date {cutoff_str}...")
    final_xgb = XGBoostForecaster(config).fit(train_holdout)
    final_preds = final_xgb.predict(test_holdout)
    test_holdout["predicted_demand"] = final_preds

    # Save trained model artifact
    model_save_path = os.path.join(config["data"]["processed_dir"], "xgb_model.json")
    final_xgb.save(model_save_path)
    print(f"Final production model saved to {model_save_path}")

    # Top Feature Importance
    feat_imp = final_xgb.get_feature_importance()
    print("\nTop 5 Driving Features (Gain):")
    print(feat_imp.head(5).to_string(index=False))

    # 5. Inventory Optimization via PuLP Linear Program
    print("\n--- STAGE 5: Linear Programming Inventory Optimization ---")
    items = list(test_holdout["item_id"].unique())
    stores = list(test_holdout["store_id"].unique())
    opt_cfg = config["optimization"]

    # Calculate item-level parameters
    lead_time_map = opt_cfg["default_lead_time_days"]
    holding_rate = opt_cfg["cost_parameters"]["holding_cost_annual_rate"] / 365.0
    stockout_margin_rate = opt_cfg["cost_parameters"]["stockout_margin_rate"]
    purchase_ratio = opt_cfg["cost_parameters"]["unit_cost_ratio"]

    # Build optimization inputs for one representative store (e.g. CA_1)
    rep_store = stores[0]
    store_test = test_holdout[test_holdout["store_id"] == rep_store].copy()
    store_items = list(store_test["item_id"].unique())

    demand_forecasts = {}
    actual_test_demand = {}
    unit_prices = {}
    lead_times = {}
    holding_costs = {}
    stockout_costs = {}
    purchase_costs = {}
    safety_stocks = {}
    initial_inventory = {}
    cached_forecasts = {}

    residuals_arr = np.array(all_validation_residuals)

    for item in store_items:
        it_df = store_test[store_test["item_id"] == item].sort_values("date")
        preds_list = it_df["predicted_demand"].values.tolist()
        actual_list = it_df["units_sold"].values.tolist()
        
        # Ensure exactly 28 days
        preds_list = (preds_list + [0.0]*28)[:28]
        actual_list = (actual_list + [0.0]*28)[:28]
        
        demand_forecasts[item] = preds_list
        actual_test_demand[item] = actual_list

        price = float(it_df["sell_price"].mean()) if "sell_price" in it_df.columns and it_df["sell_price"].mean() > 0 else 5.0
        unit_prices[item] = price

        dept = str(it_df["dept_id"].iloc[0]) if "dept_id" in it_df.columns else "FOODS"
        cat = "FOODS" if "FOOD" in dept else ("HOUSEHOLD" if "HOUSE" in dept else "HOBBIES")
        L = lead_time_map.get(cat, 3)
        lead_times[item] = L

        h = price * holding_rate
        p = price * stockout_margin_rate
        c = price * purchase_ratio

        holding_costs[item] = h
        stockout_costs[item] = p
        purchase_costs[item] = c

        # Safety stock & ROP computation
        mean_d = float(np.mean(preds_list))
        ss, rop = compute_safety_stock_and_rop(
            forecast_daily_mean=mean_d,
            lead_time_days=L,
            residuals=residuals_arr,
            unit_holding_cost_daily=h,
            unit_stockout_cost=p,
            service_level=opt_cfg.get("service_level", 0.95)
        )
        safety_stocks[item] = ss
        initial_inventory[item] = ss * 0.8 # Assume starting near safety buffer

        # Cache forecast for API serving
        cached_forecasts[f"{rep_store}_{item}"] = {
            "store_id": rep_store,
            "item_id": item,
            "horizon_days": 28,
            "forecast": [round(float(x), 2) for x in preds_list],
            "reorder_point": round(rop, 2),
            "order_quantity": 0.0, # Updated after LP
            "safety_stock": round(ss, 2),
            "unit_price": round(price, 2)
        }

    # Solve LP
    optimizer = InventoryLPOptimizer(
        planning_horizon=28,
        warehouse_capacity=opt_cfg["constraints"]["warehouse_capacity_units"],
        daily_budget=opt_cfg["constraints"]["daily_budget_dollars"]
    )
    lp_result = optimizer.solve(
        items=store_items,
        demand_forecast=demand_forecasts,
        unit_prices=unit_prices,
        lead_times=lead_times,
        holding_costs=holding_costs,
        stockout_costs=stockout_costs,
        purchase_costs=purchase_costs,
        initial_inventory=initial_inventory,
        safety_stocks=safety_stocks
    )
    print(f"PuLP Solver Status: {lp_result['status']} | Total Planned Cost: ${lp_result['total_cost']:.2f}")

    # Update order quantities
    for item in store_items:
        key = f"{rep_store}_{item}"
        first_day_order = lp_result["orders"][item][0] if item in lp_result["orders"] else 0.0
        cached_forecasts[key]["order_quantity"] = round(first_day_order, 2)

    # 6. Policy Simulation & Business KPI Replay
    print("\n--- STAGE 6: Simulation Replay vs Trailing-Average Baseline ---")
    # Baseline orders: simple order-up-to trailing 28-day average demand
    baseline_orders = {}
    for item in store_items:
        mean_d = float(np.mean(demand_forecasts[item]))
        L = lead_times[item]
        # Fixed periodic replenishment equal to expected lead demand
        baseline_orders[item] = [mean_d] * 28

    df_decision_comparison = compare_policies(
        test_actuals=actual_test_demand,
        lp_orders=lp_result["orders"],
        baseline_orders=baseline_orders,
        lead_times=lead_times,
        holding_costs=holding_costs,
        stockout_costs=stockout_costs,
        purchase_costs=purchase_costs,
        initial_inventory=initial_inventory
    )

    # 7. Export Delivery Artifacts (FastAPI Cache & Power BI Datasets)
    print("\n--- STAGE 7: Exporting Serving & Power BI Delivery Artifacts ---")
    # Save FastAPI stores
    with open(config["serving"]["forecast_store_path"], "w") as f:
        json.dump(cached_forecasts, f, indent=2)

    store_recommendations = {
        rep_store: [
            {
                "item_id": it,
                "store_id": rep_store,
                "forecast_total_28d": round(float(sum(demand_forecasts[it])), 2),
                "reorder_point": cached_forecasts[f"{rep_store}_{it}"]["reorder_point"],
                "recommended_order_qty": cached_forecasts[f"{rep_store}_{it}"]["order_quantity"],
                "current_inventory": round(initial_inventory[it], 2),
                "stockout_risk": "High" if initial_inventory[it] < cached_forecasts[f"{rep_store}_{it}"]["reorder_point"] else "Low",
                "unit_price": unit_prices[it]
            }
            for it in store_items
        ]
    }
    with open(config["serving"]["recommendations_store_path"], "w") as f:
        json.dump(store_recommendations, f, indent=2)

    # Export Power BI View Datasets
    pbi_dir = os.path.join(config["data"]["processed_dir"], "powerbi")
    os.makedirs(pbi_dir, exist_ok=True)

    # Page 1: Forecast vs Actuals table
    store_test_clean = store_test[["item_id", "store_id", "dept_id", "cat_id", "date", "units_sold", "predicted_demand"]].copy()
    store_test_clean.rename(columns={"units_sold": "actual_sales", "predicted_demand": "forecast_sales"}, inplace=True)
    store_test_clean.to_csv(os.path.join(pbi_dir, "powerbi_forecast_vs_actual.csv"), index=False)

    # Page 2: Segment Accuracy table (Category & Store aggregations)
    seg_summary = store_test_clean.groupby(["cat_id", "dept_id"]).apply(
        lambda g: pd.Series({
            "MAE": calc_mae(g["actual_sales"].values, g["forecast_sales"].values),
            "WAPE": calc_wape(g["actual_sales"].values, g["forecast_sales"].values),
            "Bias": calc_bias(g["actual_sales"].values, g["forecast_sales"].values),
            "Total_Actual": g["actual_sales"].sum(),
            "Total_Forecast": g["forecast_sales"].sum()
        })
    ).reset_index()
    seg_summary.to_csv(os.path.join(pbi_dir, "powerbi_segment_accuracy.csv"), index=False)

    # Page 3: Order Recommendations table
    df_recs = pd.DataFrame(store_recommendations[rep_store])
    df_recs.to_csv(os.path.join(pbi_dir, "powerbi_order_recommendations.csv"), index=False)

    print(f"Exported Power BI datasets to {pbi_dir}/")
    print(f"Exported API cache to {config['serving']['forecast_store_path']}")
    print("\nPIPELINE COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_pipeline()

