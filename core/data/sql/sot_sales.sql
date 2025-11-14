-- Criação da tabela SOT (Sales Order Transform)
DROP TABLE IF EXISTS SOT;

CREATE TABLE SOT AS
SELECT
    InvoiceNo,
    StockCode,
    Description,
    Quantity,
    UnitPrice,
    CustomerID,
    Country,
    InvoiceDate,
    (Quantity * UnitPrice) AS TotalPrice
FROM SOR
WHERE Quantity > 0 AND UnitPrice > 0 AND CustomerID IS NOT NULL;
