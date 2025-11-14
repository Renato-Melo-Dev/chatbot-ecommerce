import pandas as pd
from sqlalchemy import create_engine
import os

DB_PATH = os.path.join("data", "eCommerce.db")
engine = create_engine(f"sqlite:///{DB_PATH}")

def load_csv_to_sor():
    df = pd.read_csv("data/eCommerce.csv")
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df.to_sql("sor_sales", engine, if_exists="replace", index=False)
    print("CSV carregado no SOR ✅")

def transform_sor_to_sot():
    query = """
    SELECT InvoiceNo, StockCode, Description, Quantity,
           UnitPrice, (Quantity * UnitPrice) AS TotalPrice
    FROM sor_sales
    """
    df = pd.read_sql(query, engine)
    df.to_sql("sot_sales", engine, if_exists="replace", index=False)
    print("Transformação para SOT feita ✅")

def transform_sot_to_spec():
    query = """
    SELECT CustomerID, Country, SUM(TotalPrice) AS TotalSpent
    FROM sot_sales
    GROUP BY CustomerID, Country
    """
    df = pd.read_sql(query, engine)
    df.to_sql("spec_sales", engine, if_exists="replace", index=False)
    print("Transformação para SPEC feita ✅")
