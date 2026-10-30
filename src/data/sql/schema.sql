-- DDL schema definition for SQLite / PostgreSQL

-- 1. Unpivoted long-format sales table
CREATE TABLE IF NOT EXISTS sales_long (
    id TEXT,
    item_id TEXT,
    dept_id TEXT,
    cat_id TEXT,
    store_id TEXT,
    state_id TEXT,
    d TEXT,
    units_sold INTEGER
);

-- 2. Calendar and event table
CREATE TABLE IF NOT EXISTS calendar (
    date TEXT,
    wm_yr_wk INTEGER,
    weekday TEXT,
    wday INTEGER,
    month INTEGER,
    year INTEGER,
    d TEXT PRIMARY KEY,
    event_name_1 TEXT,
    event_type_1 TEXT,
    event_name_2 TEXT,
    event_type_2 TEXT,
    snap_CA INTEGER,
    snap_TX INTEGER,
    snap_WI INTEGER
);

-- 3. Weekly sell prices table
CREATE TABLE IF NOT EXISTS prices (
    store_id TEXT,
    item_id TEXT,
    wm_yr_wk INTEGER,
    sell_price REAL,
    PRIMARY KEY (store_id, item_id, wm_yr_wk)
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_sales_d ON sales_long(d);
CREATE INDEX IF NOT EXISTS idx_sales_item_store ON sales_long(item_id, store_id);
CREATE INDEX IF NOT EXISTS idx_calendar_d ON calendar(d);
CREATE INDEX IF NOT EXISTS idx_prices_lookup ON prices(store_id, item_id, wm_yr_wk);

