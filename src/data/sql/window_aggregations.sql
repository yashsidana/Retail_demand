-- SQL Window Functions and Aggregations for Feature Checks & EDA
-- Designed to guarantee NO target leakage by ending window frames at 1 PRECEDING

-- 1. Share of zero-sales days per item-store series (Intermittency check)
SELECT 
    item_id, 
    store_id,
    dept_id,
    cat_id,
    AVG(CASE WHEN units_sold = 0 THEN 1.0 ELSE 0.0 END) AS zero_share,
    COUNT(*) AS n_days,
    SUM(units_sold) AS total_units,
    AVG(units_sold) AS mean_daily_units
FROM fact_sales
GROUP BY item_id, store_id, dept_id, cat_id;

-- 2. Trailing 7-day average using window function
-- Crucial: ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING prevents including the current day's target!
SELECT 
    item_id, 
    store_id, 
    date, 
    units_sold,
    AVG(units_sold) OVER (
        PARTITION BY item_id, store_id
        ORDER BY date
        ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS trailing_7d_avg,
    AVG(units_sold) OVER (
        PARTITION BY item_id, store_id
        ORDER BY date
        ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
    ) AS trailing_28d_avg
FROM fact_sales;

-- 3. Revenue aggregation by item and store (Volume weighting)
SELECT 
    item_id,
    store_id,
    SUM(units_sold * COALESCE(sell_price, 0)) AS total_revenue
FROM fact_sales
GROUP BY item_id, store_id;

