DROP TABLE IF EXISTS sor_sales;

CREATE TABLE sor_sales (
    InvoiceNo TEXT,
    StockCode TEXT,
    Description TEXT,
    Quantity INTEGER,
    InvoiceDate TEXT,
    UnitPrice REAL,
    CustomerID INTEGER,
    Country TEXT,
    TotalPrice REAL,
    PRIMARY KEY (InvoiceNo, StockCode)
);
