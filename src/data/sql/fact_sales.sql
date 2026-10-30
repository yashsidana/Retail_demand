-- Join sales to calendar and prices to build the analysis fact table
-- SNAP column is mapped dynamically per state (CA, TX, WI)
-- LEFT JOIN on prices preserves items that were not on sale yet (censored demand / launch timing)
DROP TABLE IF EXISTS fact_sales;

CREATE TABLE fact_sales AS
SELECT
    s.id,
    s.item_id,
    s.store_id,
    s.dept_id,
    s.cat_id,
    s.state_id,
    c.date,
    c.wm_yr_wk,
    c.wday,
    c.month,
    c.year,
    c.d,
    c.event_name_1,
    c.event_type_1,
    CASE s.state_id 
        WHEN 'CA' THEN c.snap_CA
        WHEN 'TX' THEN c.snap_TX
        ELSE c.snap_WI 
    END AS snap,
    p.sell_price,
    s.units_sold
FROM sales_long s
JOIN calendar c ON s.d = c.d
LEFT JOIN prices p
    ON p.store_id = s.store_id
    AND p.item_id = s.item_id
    AND p.wm_yr_wk = c.wm_yr_wk;

CREATE INDEX IF NOT EXISTS idx_fact_item_store_date ON fact_sales(item_id, store_id, date);
CREATE INDEX IF NOT EXISTS idx_fact_date ON fact_sales(date);

