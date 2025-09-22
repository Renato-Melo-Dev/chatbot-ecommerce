-- Criação da tabela SPEC (Sales Spec/Agregado)
DROP TABLE IF EXISTS spec_sales;

CREATE TABLE spec_sales AS
SELECT
    CustomerID,
    Country,
    SUM(TotalPrice) AS TotalPrice
FROM SOT
GROUP BY CustomerID, Country;
