# core/data/database.py
import os
import pandas as pd
from sqlalchemy import create_engine, text

# Base do projeto (core/data)
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "ecommerce.db")
engine = create_engine(f"sqlite:///{DB_PATH}")

# Caminho da pasta SQL
SQL_DIR = os.path.join(BASE_DIR, "sql")  # core/data/sql/

def create_database_and_tables():
    """Cria o banco de dados e as tabelas a partir dos arquivos SQL."""
    os.makedirs(BASE_DIR, exist_ok=True)

    sql_files = [
        "sor_sales.sql",
        "sot_sales.sql",
        "spec_sales.sql"
    ]

    with engine.begin() as conn:
        for sql_file in sql_files:
            file_path = os.path.join(SQL_DIR, sql_file)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Arquivo SQL não encontrado: {file_path}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # permite múltiplas statements separadas por ;
                queries = [q.strip() for q in content.split(";") if q.strip()]
                for q in queries:
                    conn.execute(text(q))

        # Limpar dados das tabelas, se existirem
        for table in ["SOR", "SOT", "spec_sales"]:
            try:
                conn.execute(text(f"DELETE FROM {table}"))
            except Exception:
                pass

def insert_csv_to_sor(df: pd.DataFrame):
    """Insere CSV na tabela SOR."""
    df.to_sql("SOR", engine, if_exists="replace", index=False)

def run_etl_sor_to_sot():
    """Transforma dados de SOR para SOT."""
    query = """
    SELECT InvoiceNo, StockCode, Description, Quantity,
           UnitPrice, CustomerID, Country, InvoiceDate,
           (Quantity * UnitPrice) AS TotalPrice
    FROM SOR
    WHERE Quantity > 0 AND UnitPrice > 0 AND CustomerID IS NOT NULL
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

def run_etl_for_test_data(df_test: pd.DataFrame):
    """Prepara dados de teste com a mesma agregação do treino."""
    df_test["TotalPrice"] = df_test["Quantity"] * df_test["UnitPrice"]
    df = df_test.groupby(["CustomerID", "Country"], as_index=False)["TotalPrice"].sum()
    df.to_sql("spec_sales", engine, if_exists="replace", index=False)

def load_data(table_name: str) -> pd.DataFrame:
    """Carrega dados de uma tabela do banco."""
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)
