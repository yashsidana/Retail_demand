"""Pydantic schemas for the FastAPI service."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok", example="ok")
    model_version: str = Field(default="m5_xgb_v1", example="m5_xgb_v1")
    timestamp: str


class ForecastResponse(BaseModel):
    store_id: str
    item_id: str
    horizon_days: int = 28
    forecast: List[float] = Field(description="28-day daily demand predictions")
    reorder_point: float = Field(description="Recommended reorder point (ROP)")
    order_quantity: float = Field(description="Immediate order recommendation for period t=0")
    safety_stock: float = Field(description="Buffered safety stock units")
    unit_price: float = Field(description="Latest unit price ($)")


class RecommendationItem(BaseModel):
    item_id: str
    store_id: str
    forecast_total_28d: float
    reorder_point: float
    recommended_order_qty: float
    current_inventory: float
    stockout_risk: str # "Low", "Medium", "High"


class RecommendationResponse(BaseModel):
    store_id: str
    total_items_evaluated: int
    total_order_spend: float
    recommendations: List[RecommendationItem]


class WhatIfRequest(BaseModel):
    store_id: str
    item_id: str
    proposed_discount_pct: float = Field(ge=0.0, le=1.0, example=0.15)
    horizon_days: int = 28


class WhatIfResponse(BaseModel):
    store_id: str
    item_id: str
    baseline_forecast_total: float
    whatif_forecast_total: float
    projected_lift_pct: float
    original_price: float
    discounted_price: float

