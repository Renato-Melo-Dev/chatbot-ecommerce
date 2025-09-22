import os
import pandas as pd
from sqlalchemy import create_engine, text

DB_PATH = os.path.join("data", "ecommerce.db")
engine = create_engine(f"sqlite:///{DB_PATH}")

def create_database_and_tables():
    """Cria o banco de dados e as tabelas a partir dos SQLs."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    sql_files = ["data/sor_sales.sql", "data/sot_sales.sql", "data/spec_sales.sql"]
    with engine.begin() as conn:
        for file in sql_files:
            with open(file, "r", encoding="utf-8") as f:
                queries = f.read().split(";")
                for q in queries:
                    q = q.strip()
                    if q:
                        conn.execute(text(q))

def insert_csv_to_sor(df):
    """Insere CSV na tabela SOR."""
    df.to_sql("SOR", engine, if_exists="replace", index=False)

def run_etl_sor_to_sot():
    """Limpa e prepara os dados de SOR para SOT."""
    query = """
    SELECT InvoiceNo, StockCode, Description, Quantity,
           UnitPrice, CustomerID, Country, InvoiceDate,
           (Quantity*UnitPrice) AS TotalPrice
    FROM SOR
    WHERE Quantity>0 AND UnitPrice>0 AND CustomerID IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    df.to_sql("SOT", engine, if_exists="replace", index=False)

def run_etl_sot_to_spec_train():
    """Agrega dados de SOT para criar a base SPEC para treino."""
    query = """
    SELECT CustomerID, Country, SUM(TotalPrice) AS TotalPrice
    FROM SOT
    GROUP BY CustomerID, Country
    """
    df = pd.read_sql(query, engine)
    df.to_sql("spec_sales", engine, if_exists="replace", index=False)

def run_etl_for_test_data(df_test):
    """Prepara os dados de teste com a mesma agregação do treino."""
    df_test["TotalPrice"] = df_test["Quantity"] * df_test["UnitPrice"]
    df = df_test.groupby(["CustomerID", "Country"], as_index=False)["TotalPrice"].sum()
    df.to_sql("spec_sales", engine, if_exists="replace", index=False)

def load_data(table_name):
    """Carrega dados de uma tabela do banco."""
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)
