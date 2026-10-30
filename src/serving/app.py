"""FastAPI microservice for Retail Demand Forecasting & Inventory Optimization.

Provides RESTful endpoints:
- GET /health
- GET /forecast/{store_id}/{item_id}
- GET /recommendations/{store_id}
- POST /what-if (Promotional pricing elasticity scenario simulation)
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from .schemas import (
    HealthResponse,
    ForecastResponse,
    RecommendationResponse,
    RecommendationItem,
    WhatIfRequest,
    WhatIfResponse
)

app = FastAPI(
    title="Retail Demand Forecasting & Inventory Optimization API",
    description="Production serving API exposing 28-day demand forecasts and PuLP LP order recommendations.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory forecast and recommendation store
FORECAST_STORE: Dict[str, Dict[str, Any]] = {}
RECOMMENDATIONS_STORE: Dict[str, List[Dict[str, Any]]] = {}

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FORECAST_FILE = os.path.join(BASE_DIR, "data", "processed", "latest_forecasts.json")
RECOMMENDATIONS_FILE = os.path.join(BASE_DIR, "data", "processed", "latest_recommendations.json")


def load_cached_artifacts():
    """Loads precomputed batch forecasts into memory for low-latency lookup."""
    global FORECAST_STORE, RECOMMENDATIONS_STORE

    if os.path.exists(FORECAST_FILE):
        try:
            with open(FORECAST_FILE, "r") as f:
                FORECAST_STORE = json.load(f)
            print(f"Loaded {len(FORECAST_STORE)} cached forecasts into memory.")
        except Exception as e:
            print(f"Could not load forecasts cache: {e}")

    if os.path.exists(RECOMMENDATIONS_FILE):
        try:
            with open(RECOMMENDATIONS_FILE, "r") as f:
                RECOMMENDATIONS_STORE = json.load(f)
            print(f"Loaded {len(RECOMMENDATIONS_STORE)} cached store recommendations.")
        except Exception as e:
            print(f"Could not load recommendations cache: {e}")


@app.on_event("startup")
def startup_event():
    load_cached_artifacts()


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint validating service status."""
    return HealthResponse(
        status="ok",
        model_version="m5_xgb_production_v1",
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/forecast/{store_id}/{item_id}", response_model=ForecastResponse)
def get_forecast(store_id: str, item_id: str):
    """Retrieves 28-day daily forecast and inventory ordering parameters for an item-store pair."""
    key = f"{store_id}_{item_id}"
    
    if key in FORECAST_STORE:
        data = FORECAST_STORE[key]
        return ForecastResponse(**data)
        
    # If not in cache, check if generic sample exists or return 404
    raise HTTPException(
        status_code=404,
        detail=f"Forecast for series '{store_id}' and '{item_id}' not found. Please run the forecasting pipeline."
    )


@app.get("/recommendations/{store_id}", response_model=RecommendationResponse)
def get_recommendations(store_id: str):
    """Retrieves inventory replenishment recommendations for all items in a store."""
    if store_id not in RECOMMENDATIONS_STORE:
        # Check if any store recommendations exist
        if not RECOMMENDATIONS_STORE:
            raise HTTPException(
                status_code=404,
                detail="No recommendations generated yet. Run python src/pipeline.py first."
            )
        raise HTTPException(
            status_code=404,
            detail=f"Store '{store_id}' not found in active recommendations."
        )

    items = RECOMMENDATIONS_STORE[store_id]
    total_spend = sum(item.get("recommended_order_qty", 0.0) * item.get("unit_price", 1.0) for item in items)

    rec_items = [
        RecommendationItem(
            item_id=it["item_id"],
            store_id=it["store_id"],
            forecast_total_28d=it["forecast_total_28d"],
            reorder_point=it["reorder_point"],
            recommended_order_qty=it["recommended_order_qty"],
            current_inventory=it["current_inventory"],
            stockout_risk=it["stockout_risk"]
        ) for it in items
    ]

    return RecommendationResponse(
        store_id=store_id,
        total_items_evaluated=len(rec_items),
        total_order_spend=round(total_spend, 2),
        recommendations=rec_items
    )


@app.post("/what-if", response_model=WhatIfResponse)
def simulate_what_if_scenario(req: WhatIfRequest):
    """Simulates the demand impact of a promotional price discount (price elasticity)."""
    key = f"{req.store_id}_{req.item_id}"
    if key not in FORECAST_STORE:
        raise HTTPException(status_code=404, detail="Item-store series not found.")

    base_data = FORECAST_STORE[key]
    base_forecast = np.array(base_data["forecast"])
    base_total = float(np.sum(base_forecast))
    orig_price = base_data.get("unit_price", 5.0)

    # Elasticity rule of thumb (typical retail price elasticity -1.5)
    elasticity = -1.5
    price_ratio = 1.0 - req.proposed_discount_pct
    demand_multiplier = (price_ratio) ** elasticity

    whatif_forecast = base_forecast * demand_multiplier
    whatif_total = float(np.sum(whatif_forecast))
    lift_pct = ((whatif_total - base_total) / max(0.01, base_total)) * 100.0

    return WhatIfResponse(
        store_id=req.store_id,
        item_id=req.item_id,
        baseline_forecast_total=round(base_total, 2),
        whatif_forecast_total=round(whatif_total, 2),
        projected_lift_pct=round(lift_pct, 2),
        original_price=round(orig_price, 2),
        discounted_price=round(orig_price * (1.0 - req.proposed_discount_pct), 2)
    )

