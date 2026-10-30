"""Unit test for Linear Programming Inventory Optimization with PuLP.

Constructs a 2-item, 2-period toy problem with known analytical properties,
asserting inventory balance equations, capacity bounds, and non-negativity.
"""

import pytest
import numpy as np
from src.optimization.inventory_lp import InventoryLPOptimizer


def test_two_item_lp_toy_problem():
    """Validates that PuLP finds the optimal solution on a 2-item toy scenario."""
    items = ["ITEM_A", "ITEM_B"]
    planning_horizon = 3 # 3 days

    demand_forecast = {
        "ITEM_A": [10.0, 10.0, 10.0],
        "ITEM_B": [5.0, 5.0, 5.0]
    }
    unit_prices = {"ITEM_A": 10.0, "ITEM_B": 20.0}
    lead_times = {"ITEM_A": 0, "ITEM_B": 0}
    holding_costs = {"ITEM_A": 0.5, "ITEM_B": 1.0}
    stockout_costs = {"ITEM_A": 15.0, "ITEM_B": 30.0}
    purchase_costs = {"ITEM_A": 5.0, "ITEM_B": 10.0}
    initial_inventory = {"ITEM_A": 0.0, "ITEM_B": 0.0}
    safety_stocks = {"ITEM_A": 2.0, "ITEM_B": 1.0}

    optimizer = InventoryLPOptimizer(
        planning_horizon=planning_horizon,
        warehouse_capacity=1000.0,
        daily_budget=5000.0
    )

    result = optimizer.solve(
        items=items,
        demand_forecast=demand_forecast,
        unit_prices=unit_prices,
        lead_times=lead_times,
        holding_costs=holding_costs,
        stockout_costs=stockout_costs,
        purchase_costs=purchase_costs,
        initial_inventory=initial_inventory,
        safety_stocks=safety_stocks
    )

    assert result["status"] == "Optimal", f"Solver failed with status {result['status']}"
    assert result["total_cost"] is not None

    orders = result["orders"]
    inventory = result["inventory"]
    unmet = result["unmet_demand"]

    # 1. Non-negativity check
    for item in items:
        for t in range(planning_horizon):
            assert orders[item][t] >= 0.0
            assert inventory[item][t] >= 0.0
            assert unmet[item][t] >= 0.0

    # 2. Safety stock satisfaction
    for item in items:
        for t in range(planning_horizon):
            assert inventory[item][t] >= safety_stocks[item] - 1e-4

    # 3. Inventory balance verification:
    # I_t = I_{t-1} + Q_t - D_t + U_t
    for item in items:
        prev = initial_inventory[item]
        for t in range(planning_horizon):
            expected_inv = prev + orders[item][t] - demand_forecast[item][t] + unmet[item][t]
            assert np.isclose(inventory[item][t], expected_inv, atol=1e-3)
            prev = inventory[item][t]


def test_capacity_constraint_enforcement():
    """Checks that when budget is restricted, solver respects the cap."""
    items = ["ITEM_A"]
    demand_forecast = {"ITEM_A": [50.0, 50.0]}
    unit_prices = {"ITEM_A": 10.0}
    lead_times = {"ITEM_A": 0}
    holding_costs = {"ITEM_A": 1.0}
    stockout_costs = {"ITEM_A": 20.0}
    purchase_costs = {"ITEM_A": 5.0} # $5 per unit
    initial_inventory = {"ITEM_A": 0.0}
    safety_stocks = {"ITEM_A": 0.0}

    # Strict daily budget of $100 -> maximum order of 20 units
    daily_budget = 100.0
    optimizer = InventoryLPOptimizer(planning_horizon=2, daily_budget=daily_budget)

    result = optimizer.solve(
        items=items,
        demand_forecast=demand_forecast,
        unit_prices=unit_prices,
        lead_times=lead_times,
        holding_costs=holding_costs,
        stockout_costs=stockout_costs,
        purchase_costs=purchase_costs,
        initial_inventory=initial_inventory,
        safety_stocks=safety_stocks
    )

    assert result["status"] == "Optimal"
    for t in range(2):
        spend = result["orders"]["ITEM_A"][t] * purchase_costs["ITEM_A"]
        assert spend <= daily_budget + 1e-4

