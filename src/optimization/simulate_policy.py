"""Discrete-event inventory policy simulation and business KPI benchmarking.

Replays ordering policies day-by-day against realized test demand:
1. Proposed LP Policy (guided by global XGBoost forecasts & safety stock)
2. Heuristic Baseline Policy (order-up-to trailing 28-day average demand)

Evaluates: Total Cost, Fill Rate (Service Level), Stockout Frequency, and Average Inventory.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any


def simulate_single_series(
    actual_demand: List[float],
    orders: List[float],
    lead_time: int,
    initial_inventory: float,
    holding_cost_daily: float,
    stockout_penalty: float,
    purchase_cost: float
) -> Dict[str, float]:
    """Simulates a single item-store series day-by-day over the horizon."""
    n_days = len(actual_demand)
    inv = initial_inventory
    pipeline = [0.0] * lead_time # in-transit queue

    total_holding_cost = 0.0
    total_stockout_cost = 0.0
    total_purchase_cost = 0.0
    total_demand = 0.0
    total_served = 0.0
    stockout_days = 0
    inv_history = []

    for t in range(n_days):
        # 1. Receive incoming shipments
        arrived = pipeline.pop(0) if pipeline else 0.0
        inv += arrived

        # 2. Place new order
        order_qty = orders[t] if t < len(orders) else 0.0
        pipeline.append(order_qty)
        total_purchase_cost += order_qty * purchase_cost

        # 3. Realize customer demand
        demand = actual_demand[t]
        total_demand += demand

        # 4. Fulfill demand
        served = min(inv, demand)
        lost = demand - served
        inv -= served

        total_served += served
        if lost > 0:
            stockout_days += 1
            total_stockout_cost += lost * stockout_penalty

        # 5. Holding cost on remaining inventory
        total_holding_cost += inv * holding_cost_daily
        inv_history.append(inv)

    fill_rate = total_served / (total_demand + 1e-6)
    stockout_freq = stockout_days / max(1, n_days)
    avg_inventory = float(np.mean(inv_history))
    total_cost = total_holding_cost + total_stockout_cost + total_purchase_cost

    return {
        "total_cost": total_cost,
        "holding_cost": total_holding_cost,
        "stockout_cost": total_stockout_cost,
        "purchase_cost": total_purchase_cost,
        "fill_rate": fill_rate,
        "stockout_frequency": stockout_freq,
        "average_inventory": avg_inventory,
        "total_demand": total_demand,
        "total_served": total_served
    }


def simulate_inventory_policy(
    test_actuals: Dict[str, List[float]],
    policy_orders: Dict[str, List[float]],
    lead_times: Dict[str, int],
    holding_costs: Dict[str, float],
    stockout_costs: Dict[str, float],
    purchase_costs: Dict[str, float],
    initial_inventory: Dict[str, float]
) -> Dict[str, Any]:
    """Runs simulation across all items and aggregates business KPIs."""
    item_results = {}
    total_cost = 0.0
    total_holding = 0.0
    total_stockout = 0.0
    total_purchase = 0.0
    total_demand_all = 0.0
    total_served_all = 0.0
    stockout_day_counts = 0
    total_days_evaluated = 0

    for item, actual in test_actuals.items():
        orders = policy_orders.get(item, [0.0] * len(actual))
        L = lead_times.get(item, 2)
        h = holding_costs.get(item, 0.02)
        p = stockout_costs.get(item, 2.0)
        c = purchase_costs.get(item, 5.0)
        init_inv = initial_inventory.get(item, 0.0)

        res = simulate_single_series(
            actual_demand=actual,
            orders=orders,
            lead_time=L,
            initial_inventory=init_inv,
            holding_cost_daily=h,
            stockout_penalty=p,
            purchase_cost=c
        )
        item_results[item] = res
        total_cost += res["total_cost"]
        total_holding += res["holding_cost"]
        total_stockout += res["stockout_cost"]
        total_purchase += res["purchase_cost"]
        total_demand_all += res["total_demand"]
        total_served_all += res["total_served"]
        stockout_day_counts += res["stockout_frequency"] * len(actual)
        total_days_evaluated += len(actual)

    aggregated = {
        "total_cost": total_cost,
        "holding_cost": total_holding,
        "stockout_cost": total_stockout,
        "purchase_cost": total_purchase,
        "overall_fill_rate": total_served_all / (total_demand_all + 1e-6),
        "overall_stockout_frequency": stockout_day_counts / max(1, total_days_evaluated),
        "item_details": item_results
    }
    return aggregated


def compare_policies(
    test_actuals: Dict[str, List[float]],
    lp_orders: Dict[str, List[float]],
    baseline_orders: Dict[str, List[float]],
    lead_times: Dict[str, int],
    holding_costs: Dict[str, float],
    stockout_costs: Dict[str, float],
    purchase_costs: Dict[str, float],
    initial_inventory: Dict[str, float]
) -> pd.DataFrame:
    """Produces the business-level comparison table featured in the interview guide."""
    lp_res = simulate_inventory_policy(
        test_actuals, lp_orders, lead_times, holding_costs, stockout_costs, purchase_costs, initial_inventory
    )
    base_res = simulate_inventory_policy(
        test_actuals, baseline_orders, lead_times, holding_costs, stockout_costs, purchase_costs, initial_inventory
    )

    df_comp = pd.DataFrame([
        {
            "Policy": "Trailing-average order-up-to (Baseline)",
            "Total Cost ($)": round(base_res["total_cost"], 2),
            "Fill Rate (%)": round(base_res["overall_fill_rate"] * 100, 2),
            "Stockout Frequency (%)": round(base_res["overall_stockout_frequency"] * 100, 2),
            "Holding Cost ($)": round(base_res["holding_cost"], 2),
            "Stockout Penalty ($)": round(base_res["stockout_cost"], 2)
        },
        {
            "Policy": "LP plan with XGBoost forecast",
            "Total Cost ($)": round(lp_res["total_cost"], 2),
            "Fill Rate (%)": round(lp_res["overall_fill_rate"] * 100, 2),
            "Stockout Frequency (%)": round(lp_res["overall_stockout_frequency"] * 100, 2),
            "Holding Cost ($)": round(lp_res["holding_cost"], 2),
            "Stockout Penalty ($)": round(lp_res["stockout_cost"], 2)
        }
    ])

    cost_diff = (lp_res["total_cost"] - base_res["total_cost"]) / (base_res["total_cost"] + 1e-6) * 100
    fill_diff = (lp_res["overall_fill_rate"] - base_res["overall_fill_rate"]) * 100

    print("\n================== DECISION LAYER RESULTS ==================")
    print(df_comp.to_string(index=False))
    print(f"Cost reduction: {-cost_diff:.2f}% | Fill rate change: {fill_diff:+.2f} percentage points")
    print("============================================================\n")

    return df_comp

