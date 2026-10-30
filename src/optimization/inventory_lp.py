"""Linear Programming Inventory Optimizer with PuLP.

Formulates a multi-item, multi-period inventory replenishment problem minimizing
total holding, stockout (unmet demand penalty), and purchasing costs subject to:
- Inventory balance equations with deterministic lead times
- Safety stock lower bounds (derived from forecast uncertainty & service levels)
- Warehouse storage volume capacities
- Periodic purchasing budget constraints
"""

import numpy as np
import pandas as pd
import pulp
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional


def compute_safety_stock_and_rop(
    forecast_daily_mean: float,
    lead_time_days: int,
    residuals: np.ndarray,
    unit_holding_cost_daily: float,
    unit_stockout_cost: float,
    service_level: Optional[float] = None
) -> Tuple[float, float]:
    """Computes Safety Stock (SS) and Reorder Point (ROP).
    
    Uses the Newsvendor critical ratio:
        alpha* = C_u / (C_u + C_o)
    where C_u is underage cost (stockout penalty) and C_o is overage cost (holding cost).
    
    If residuals are skewed/intermittent, calculates empirical quantile of validation errors;
    otherwise uses normal approximation: SS = z_alpha * sigma_lead_time.
    """
    if service_level is None:
        # Critical ratio
        service_level = unit_stockout_cost / (unit_stockout_cost + unit_holding_cost_daily + 1e-6)
        service_level = float(np.clip(service_level, 0.50, 0.99))

    # Lead time forecast error std
    if len(residuals) > 5:
        # Empirical quantile or std
        sigma_daily = float(np.std(residuals))
        sigma_lead = sigma_daily * np.sqrt(lead_time_days)
        z = float(norm.ppf(service_level))
        safety_stock = max(0.0, z * sigma_lead)
    else:
        # Fallback heuristic: 30% of lead-time demand
        safety_stock = 0.3 * forecast_daily_mean * lead_time_days

    # ROP = Expected demand during lead time + Safety Stock
    expected_lead_demand = forecast_daily_mean * lead_time_days
    rop = expected_lead_demand + safety_stock

    return float(safety_stock), float(rop)


class InventoryLPOptimizer:
    """Solves the multi-item multi-period linear program for inventory planning."""

    def __init__(
        self,
        planning_horizon: int = 28,
        warehouse_capacity: float = 50000.0,
        daily_budget: float = 20000.0
    ):
        self.planning_horizon = planning_horizon
        self.warehouse_capacity = warehouse_capacity
        self.daily_budget = daily_budget

    def solve(
        self,
        items: List[str],
        demand_forecast: Dict[str, List[float]],
        unit_prices: Dict[str, float],
        lead_times: Dict[str, int],
        holding_costs: Dict[str, float],
        stockout_costs: Dict[str, float],
        purchase_costs: Dict[str, float],
        initial_inventory: Dict[str, float],
        safety_stocks: Dict[str, float],
        unit_volumes: Optional[Dict[str, float]] = None,
        max_order_quantities: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Solves the linear program using CBC / PuLP.
        
        Returns:
            Dictionary containing optimization status, total cost, and optimal trajectories
            for order quantities Q[i][t], inventory levels I[i][t], and unmet demand U[i][t].
        """
        periods = list(range(self.planning_horizon))
        
        if unit_volumes is None:
            unit_volumes = {i: 1.0 for i in items}
        if max_order_quantities is None:
            max_order_quantities = {i: 1000.0 for i in items}

        # Initialize LP problem
        prob = pulp.LpProblem("retail_inventory_optimization", pulp.LpMinimize)

        # Decision variables
        Q = pulp.LpVariable.dicts("order", (items, periods), lowBound=0)
        I = pulp.LpVariable.dicts("inv", (items, periods), lowBound=0)
        U = pulp.LpVariable.dicts("lost", (items, periods), lowBound=0)

        # Objective Function: Minimize holding + stockout penalty + purchase costs
        prob += pulp.lpSum(
            holding_costs[i] * I[i][t] +
            stockout_costs[i] * U[i][t] +
            purchase_costs[i] * Q[i][t]
            for i in items for t in periods
        )

        # Constraints
        for i in items:
            L = lead_times.get(i, 2)
            for t in periods:
                prev_inv = I[i][t - 1] if t > 0 else initial_inventory.get(i, 0.0)
                arrived_order = Q[i][t - L] if t - L >= 0 else 0.0
                d_hat = demand_forecast[i][t]

                # 1. Inventory Balance Constraint: I_t = I_{t-1} + Q_{t-L} - D_hat_t + U_t
                prob += I[i][t] == prev_inv + arrived_order - d_hat + U[i][t]

                # 2. Safety stock lower bound
                prob += I[i][t] >= safety_stocks.get(i, 0.0)

                # 3. Unmet demand cannot exceed forecast demand
                prob += U[i][t] <= d_hat

                # 4. Maximum supplier order quantity
                prob += Q[i][t] <= max_order_quantities.get(i, 1000.0)

        # Shared Resource Constraints per period
        for t in periods:
            # 5. Total warehouse space capacity constraint
            prob += pulp.lpSum(unit_volumes[i] * I[i][t] for i in items) <= self.warehouse_capacity

            # 6. Periodic purchasing budget limit
            prob += pulp.lpSum(purchase_costs[i] * Q[i][t] for i in items) <= self.daily_budget

        # Solve with CBC
        solver = pulp.PULP_CBC_CMD(msg=False)
        status = prob.solve(solver)
        status_str = pulp.LpStatus[status]

        # Extract optimal values
        orders_opt = {i: [float(pulp.value(Q[i][t])) for t in periods] for i in items}
        inventory_opt = {i: [float(pulp.value(I[i][t])) for t in periods] for i in items}
        unmet_opt = {i: [float(pulp.value(U[i][t])) for t in periods] for i in items}

        return {
            "status": status_str,
            "total_cost": float(pulp.value(prob.objective)) if status == pulp.LpStatusOptimal else None,
            "orders": orders_opt,
            "inventory": inventory_opt,
            "unmet_demand": unmet_opt
        }

