DROP TABLE IF EXISTS spec_sales;

CREATE TABLE spec_sales (
    column_name TEXT,
    data_type TEXT,
    is_target INTEGER,  -- 0 = False, 1 = True
    description TEXT
);
