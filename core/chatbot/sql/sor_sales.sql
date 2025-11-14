-- Criação da tabela SOR (Sales Order Raw)
DROP TABLE IF EXISTS SOR;

CREATE TABLE SOR (
    InvoiceNo TEXT,
    StockCode TEXT,
    Description TEXT,
    Quantity INTEGER,
    InvoiceDate TEXT,
    UnitPrice REAL,
    CustomerID INTEGER,
    Country TEXT
);
